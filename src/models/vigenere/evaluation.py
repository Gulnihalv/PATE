"""Vigenere evaluation harness for the neural models and the classical solver.

The primary condition is the full pipeline: each solver estimates the period
from the ciphertext itself and then decrypts under its own estimate, which is
the comparison the results are built on. A secondary condition supplies the
true period to both solvers, which isolates shift recovery from period
recovery and is reported separately.

Field naming follows that split. Unprefixed fields are the pipeline:

  char_acc            character accuracy end to end
  key_acc             period recovered exactly
  exact_key           full key recovered -- period and every shift
  key_sym_acc         per-symbol key accuracy, a wrong period counting as a
                      total miss
  consistent_char_acc decryption re-derived from the coset-majority key
  avg_ms              mean wall-clock latency

`oracle_`-prefixed fields are the same quantities with the true period given:

  oracle_char_acc, oracle_exact_key, oracle_key_sym_acc

`run_neural_eval`     : the grid for a Vigenere model.
`run_classical_eval`  : the same grid for the classical Kasiski + coincidence +
                        chi-square solver.
`blind_period_acc`    : per-key-length accuracy of the unconditioned first-pass
                        estimate, for models that have a key-length head.
`blind_period_grid`   : the same estimate per (key length, text length) cell.
`no_fallback_char_grid`: character accuracy with the fallback disabled.
`print_final_table`   : PATE (blind, pipeline, OOD) beside the classical solver.
`load_clean_text`     : read a corpus and drop every character outside the alphabet.
`frequency_distance`  : L1 distance between the character distributions of two
                        corpora, used to confirm that an out-of-distribution set
                        really differs from the validation set.
`run_ood_eval`        : character accuracy and blind period accuracy on an
                        out-of-distribution corpus, reported next to the
                        frequency distance.

All grid functions return the same structured dict, so tables and heatmaps read
from the same data.
"""

import random
import time
from collections import Counter

import torch

from models.vigenere import classical as vcl
from models.vigenere import inference as vinf


def get_fixed_key(key_len, seed_val, alphabet_chars):
    """Deterministic Vigenere key (seed = key_len*10000 + sample index)."""
    random.seed(key_len * 10000 + seed_val)
    return "".join(random.choices(alphabet_chars, k=key_len))


def _clean(text, char2idx):
    return "".join(c for c in text.lower() if c in char2idx)


def _alpha_pos(alphabet_chars):
    return {c: i for i, c in enumerate(alphabet_chars)}


def load_clean_text(path, alphabet_chars):
    """Read a corpus and keep only characters that belong to the alphabet."""
    char2idx, _ = vinf.build_vocab(alphabet_chars)
    with open(path, encoding="utf-8") as f:
        raw = f.read().replace("\n", " ")
    return _clean(raw, char2idx)


def frequency_distance(text_a, text_b, alphabet_chars):
    """L1 distance between the character distributions of two corpora.

    A value near zero means the two corpora share a distribution, which makes an
    out-of-distribution comparison uninformative. Values above roughly 0.05
    indicate a clear difference in register or genre.
    """
    def dist(text):
        counts = Counter(text)
        total = sum(counts.values()) or 1
        return [counts.get(c, 0) / total for c in alphabet_chars]

    fa, fb = dist(text_a), dist(text_b)
    return sum(abs(x - y) for x, y in zip(fa, fb))


# --------------------------------------------------------------------------- #
# key-level metrics
# --------------------------------------------------------------------------- #

def recover_key_shifts(cipher, plain, key_len, alphabet_chars):
    """Read the key off a decoding by majority vote within each coset.

    Every position i of a Vigenere ciphertext is shifted by the same key symbol
    as every other position congruent to i modulo d, so a decoding implies one
    shift vote per character. Taking the majority within each coset enforces
    that constraint, which a per-character decoder is free to violate.
    """
    pos = _alpha_pos(alphabet_chars)
    n_alpha = len(alphabet_chars)
    votes = [Counter() for _ in range(key_len)]
    for i in range(min(len(cipher), len(plain))):
        c, p = cipher[i], plain[i]
        if c in pos and p in pos:
            votes[i % key_len][(pos[c] - pos[p]) % n_alpha] += 1
    return [v.most_common(1)[0][0] if v else 0 for v in votes]


def apply_key_shifts(cipher, shifts, alphabet_chars):
    """Decrypt a ciphertext under an explicit shift vector."""
    pos = _alpha_pos(alphabet_chars)
    n_alpha = len(alphabet_chars)
    d = len(shifts)
    out = []
    for i, c in enumerate(cipher):
        if c in pos:
            out.append(alphabet_chars[(pos[c] - shifts[i % d]) % n_alpha])
        else:
            out.append(c)
    return "".join(out)


def _key_metrics(cipher, decoded, true_key, pred_key_len, alphabet_chars):
    """Exact-key, per-symbol-key and coset-consistent character accuracy.

    A wrong period makes the key uncomparable, so such a sample counts as a
    total key miss; the period accuracy is reported separately and the two
    together say whether a failure came from the period or from the shifts.
    """
    pos = _alpha_pos(alphabet_chars)
    true_shifts = [pos[c] for c in true_key]
    n_key = len(true_shifts)

    if pred_key_len != n_key:
        return 0, 0, n_key, None

    shifts = recover_key_shifts(cipher, decoded, pred_key_len, alphabet_chars)
    sym_hits = sum(a == b for a, b in zip(shifts, true_shifts))
    exact = int(sym_hits == n_key)
    consistent = apply_key_shifts(cipher, shifts, alphabet_chars)
    return exact, sym_hits, n_key, consistent


# --------------------------------------------------------------------------- #
# neural grid
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _oracle_logits(model, src_list, key_len, device, batch_size=64):
    """Second pass only: decrypt with the true period supplied.

    The counterpart of the classical oracle condition. The model's own period
    estimate is bypassed, so what remains is its shift recovery.
    """
    preds = []
    for i in range(0, len(src_list), batch_size):
        chunk = torch.stack(src_list[i:i + batch_size])
        kl = torch.full((chunk.size(0),), key_len, dtype=torch.long, device=device)
        plain_logits, _, _ = model(chunk, kl)
        preds.append(plain_logits.argmax(dim=-1).cpu())
    return torch.cat(preds)


def run_neural_eval(model, validation_text, test_lengths, test_key_lens,
                    alphabet_chars, num_samples=500, device="cpu",
                    oracle=True, key_metrics=True):
    """Per-cell accuracy, key recovery and timing for a Vigenere model.

    The pipeline fields are the primary result; `oracle=True` adds the
    secondary condition in which the true period is supplied.
    """
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    full_clean = _clean(validation_text, char2idx)

    results = {}
    for key_len in test_key_lens:
        results[key_len] = {}
        for chunk_len in test_lengths:
            total_chars = correct_chars = correct_key_len = 0
            cons_total = cons_correct = 0
            exact_keys = key_sym_hits = key_sym_total = 0
            total_ms = fallbacks = 0.0
            srcs, ciphers, plains, keys = [], [], [], []

            for i in range(num_samples):
                start = (i * chunk_len * 13) % max(1, len(full_clean) - chunk_len)
                plain = full_clean[start:start + chunk_len]
                key = get_fixed_key(key_len, i, alphabet_chars)
                cipher, clean_plain = vinf.encrypt_vigenere(plain, key, char2idx, idx2char)
                src = vinf.text_to_tensor(cipher, char2idx, device)

                t0 = time.perf_counter()
                result = model.decode(src)
                total_ms += (time.perf_counter() - t0) * 1000

                decoded = "".join(idx2char.get(idx, "") for idx in result["plaintext"].tolist())
                m = min(len(clean_plain), len(decoded))
                correct_chars += sum(a == b for a, b in zip(clean_plain[:m], decoded[:m]))
                total_chars += m
                pred_key_len = int(result["best_key_len"])
                correct_key_len += int(pred_key_len == key_len)
                fallbacks += int(result.get("method") == "fallback_bruteforce")

                if key_metrics:
                    exact, sym_hits, n_key, consistent = _key_metrics(
                        cipher, decoded, key, pred_key_len, alphabet_chars)
                    exact_keys += exact
                    key_sym_hits += sym_hits
                    key_sym_total += n_key
                    ref = consistent if consistent is not None else decoded
                    cm = min(len(clean_plain), len(ref))
                    cons_correct += sum(a == b for a, b in zip(clean_plain[:cm], ref[:cm]))
                    cons_total += cm

                srcs.append(src.squeeze(0) if src.dim() > 1 else src)
                ciphers.append(cipher)
                plains.append(clean_plain)
                keys.append(key)

            cell = {
                "char_acc": correct_chars / max(total_chars, 1) * 100,
                "key_acc": correct_key_len / num_samples * 100,
                "avg_ms": total_ms / num_samples,
                "fb_rate": fallbacks / num_samples * 100,
            }

            if key_metrics:
                cell["exact_key"] = exact_keys / num_samples * 100
                cell["key_sym_acc"] = key_sym_hits / max(key_sym_total, 1) * 100
                cell["consistent_char_acc"] = cons_correct / max(cons_total, 1) * 100

            if oracle:
                pred = _oracle_logits(model, srcs, key_len, device)
                o_correct = o_total = 0
                o_exact = o_sym = o_sym_total = 0
                for row, ref, ct, kk in zip(pred, plains, ciphers, keys):
                    dec = "".join(idx2char.get(idx, "") for idx in row.tolist())
                    m = min(len(ref), len(dec))
                    o_correct += sum(a == b for a, b in zip(ref[:m], dec[:m]))
                    o_total += m
                    if key_metrics:
                        e, s, n_key, _ = _key_metrics(ct, dec, kk, key_len, alphabet_chars)
                        o_exact += e
                        o_sym += s
                        o_sym_total += n_key
                cell["oracle_char_acc"] = o_correct / max(o_total, 1) * 100
                if key_metrics:
                    cell["oracle_exact_key"] = o_exact / num_samples * 100
                    cell["oracle_key_sym_acc"] = o_sym / max(o_sym_total, 1) * 100

            results[key_len][chunk_len] = cell
    return results


# --------------------------------------------------------------------------- #
# classical grid
# --------------------------------------------------------------------------- #

def run_classical_eval(validation_text, test_lengths, test_key_lens,
                       alphabet_chars, num_samples=500, min_k=3, max_k=32,
                       tol=vcl.FUNDAMENTAL_TOL, oracle=True):
    """Classical Vigenere baseline over the same grid and the same conditions.

    `min_k` and `max_k` bound the period search and must cover the experimental
    key-length range; `searched_max_k` records the bound actually reached, which
    is lower than max_k on texts too short to leave two characters per coset.
    """
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    full_clean = _clean(validation_text, char2idx)
    pos = _alpha_pos(alphabet_chars)

    results = {}
    for key_len in test_key_lens:
        results[key_len] = {}
        for chunk_len in test_lengths:
            total_chars = correct_chars = correct_key_len = 0
            exact_key = key_sym_hits = key_sym_total = 0
            oracle_total = oracle_correct = 0
            oracle_exact = oracle_sym = 0
            total_ms = 0.0
            searched_max_k = 0

            for i in range(num_samples):
                start = (i * chunk_len * 13) % max(1, len(full_clean) - chunk_len)
                plain = full_clean[start:start + chunk_len]
                key = get_fixed_key(key_len, i, alphabet_chars)
                cipher, clean_plain = vinf.encrypt_vigenere(plain, key, char2idx, idx2char)
                searched_max_k = vcl.searchable_range(cipher, min_k, max_k)[1]
                true_shifts = [pos[c] for c in key]

                # --- pipeline: the solver estimates the period itself ---
                t0 = time.perf_counter()
                decoded, pred_k, p_shifts = vcl.classical_vigenere_solve(
                    cipher, min_k, max_k, tol=tol)
                total_ms += (time.perf_counter() - t0) * 1000

                m = min(len(clean_plain), len(decoded))
                correct_chars += sum(a == b for a, b in zip(clean_plain[:m], decoded[:m]))
                total_chars += m
                correct_key_len += int(pred_k == key_len)

                p_hits = (sum(a == b for a, b in zip(p_shifts, true_shifts))
                          if pred_k == len(true_shifts) else 0)
                key_sym_hits += p_hits
                key_sym_total += len(true_shifts)
                exact_key += int(p_hits == len(true_shifts))

                # --- oracle: the true period is supplied ---
                if oracle:
                    o_decoded, _, o_shifts = vcl.classical_vigenere_solve(
                        cipher, min_k, max_k, true_key_len=key_len, tol=tol)
                    om = min(len(clean_plain), len(o_decoded))
                    oracle_correct += sum(a == b for a, b in zip(clean_plain[:om],
                                                                 o_decoded[:om]))
                    oracle_total += om
                    o_hits = sum(a == b for a, b in zip(o_shifts, true_shifts))
                    oracle_sym += o_hits
                    oracle_exact += int(o_hits == len(true_shifts))

            cell = {
                "char_acc":       correct_chars / max(total_chars, 1) * 100,
                "key_acc":        correct_key_len / num_samples * 100,
                "exact_key":      exact_key / num_samples * 100,
                "key_sym_acc":    key_sym_hits / max(key_sym_total, 1) * 100,
                "avg_ms":         total_ms / num_samples,
                "searched_max_k": searched_max_k,
            }
            if oracle:
                cell["oracle_char_acc"] = oracle_correct / max(oracle_total, 1) * 100
                cell["oracle_exact_key"] = oracle_exact / num_samples * 100
                cell["oracle_key_sym_acc"] = oracle_sym / max(key_sym_total, 1) * 100

            results[key_len][chunk_len] = cell
    return results


@torch.no_grad()
def blind_period_acc(model, text, alphabet_chars, text_len=512, num_samples=300,
                     device="cpu", unknown_key=0):
    """Per-key-length accuracy of the unconditioned first-pass estimate.

    Only applies to models with a key-length head; the baselines return no
    period and are not evaluated here.
    """
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    clean = _clean(text, char2idx)

    acc = {}
    for k in range(model.min_key_len, model.max_key_len + 1):
        batch = []
        for i in range(num_samples):
            start = (i * text_len * 13) % max(1, len(clean) - text_len)
            key = get_fixed_key(k, i, alphabet_chars)
            cipher, _ = vinf.encrypt_vigenere(clean[start:start + text_len],
                                              key, char2idx, idx2char)
            batch.append(vinf.text_to_tensor(cipher, char2idx, device))
        src = torch.stack(batch)
        unknown = torch.full((num_samples,), unknown_key, dtype=torch.long, device=device)
        _, _, keylen_logits = model(src, unknown)
        pred = keylen_logits.argmax(dim=-1) + model.min_key_len
        acc[k] = (pred == k).float().mean().item() * 100
    return acc


@torch.no_grad()
def blind_period_grid(model, text, test_lengths, test_key_lens, alphabet_chars,
                      num_samples=500, device="cpu", unknown_key=0, batch_size=100):
    """Blind period accuracy for every (key_len, text_len) cell.

    The first-pass estimate with no validity safeguard: the same samples as
    `run_neural_eval`, so the two grids line up cell for cell.
    """
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    clean = _clean(text, char2idx)

    out = {}
    for k in test_key_lens:
        out[k] = {}
        for L in test_lengths:
            hits = 0
            for b0 in range(0, num_samples, batch_size):
                batch = []
                for i in range(b0, min(b0 + batch_size, num_samples)):
                    start = (i * L * 13) % max(1, len(clean) - L)
                    key = get_fixed_key(k, i, alphabet_chars)
                    cipher, _ = vinf.encrypt_vigenere(clean[start:start + L],
                                                      key, char2idx, idx2char)
                    batch.append(vinf.text_to_tensor(cipher, char2idx, device).reshape(-1))
                src = torch.stack(batch)
                unknown = torch.full((src.size(0),), unknown_key,
                                     dtype=torch.long, device=device)
                _, _, keylen_logits = model(src, unknown)
                pred = keylen_logits.argmax(dim=-1) + model.min_key_len
                hits += int((pred == k).sum().item())
            out[k][L] = hits / num_samples * 100
    return out


@torch.no_grad()
def no_fallback_char_grid(model, text, test_lengths, test_key_lens, alphabet_chars,
                          num_samples=500, device="cpu"):
    """Character accuracy with the validity safeguard disabled.

    Decryption proceeds from the blind period estimate whether or not it is
    correct, on the same samples as `run_neural_eval`.
    """
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    clean = _clean(text, char2idx)

    out = {}
    for k in test_key_lens:
        out[k] = {}
        for L in test_lengths:
            correct = total = 0
            for i in range(num_samples):
                start = (i * L * 13) % max(1, len(clean) - L)
                key = get_fixed_key(k, i, alphabet_chars)
                cipher, plain = vinf.encrypt_vigenere(clean[start:start + L],
                                                      key, char2idx, idx2char)
                src = vinf.text_to_tensor(cipher, char2idx, device)
                r = model.decode(src, use_fallback=False)
                dec = "".join(idx2char.get(x, "") for x in r["plaintext"].tolist())
                m = min(len(plain), len(dec))
                correct += sum(a == b for a, b in zip(plain[:m], dec[:m]))
                total += m
            out[k][L] = correct / max(total, 1) * 100
    return out


def run_ood_eval(model, ood_path, validation_path, alphabet_chars,
                 test_lengths=(128, 256, 512), test_key_lens=None,
                 num_samples=500, device="cpu", blind=True, unknown_key=0):
    """Evaluate a model on an out-of-distribution corpus.

    Reports the character-frequency distance between the two corpora alongside
    the results, so that a null result can be distinguished from a test that was
    never a test: if the distance is near zero the corpora share a distribution
    and the comparison says nothing about generalisation.

    Returns a dict with keys `freq_distance`, `grid` and, when the model has a
    key-length head, `blind_period`.
    """
    ood_text = load_clean_text(ood_path, alphabet_chars)
    val_text = load_clean_text(validation_path, alphabet_chars)

    out = {
        "freq_distance": frequency_distance(ood_text, val_text, alphabet_chars),
        "ood_chars": len(ood_text),
        "validation_chars": len(val_text),
    }

    if test_key_lens is None:
        test_key_lens = list(range(model.min_key_len, model.max_key_len + 1))

    out["grid"] = run_neural_eval(model, ood_text, list(test_lengths), list(test_key_lens),
                                  alphabet_chars, num_samples=num_samples, device=device)

    if blind and hasattr(model, "keylen_head"):
        out["blind_period"] = blind_period_acc(model, ood_text, alphabet_chars,
                                               text_len=max(test_lengths),
                                               num_samples=min(num_samples, 300),
                                               device=device, unknown_key=unknown_key)
    return out


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #

_SHORT = {
    "char_acc": "CHAR%", "key_acc": "KEY%", "exact_key": "EXKEY%",
    "key_sym_acc": "KSYM%", "consistent_char_acc": "CONS%", "avg_ms": "ms",
    "oracle_char_acc": "ORAC%", "oracle_exact_key": "O-EXKEY%",
    "oracle_key_sym_acc": "O-KSYM%", "searched_max_k": "maxk",
}


def print_grid(results, title="", fields=("char_acc", "key_acc", "avg_ms")):
    """Print the (key_len x text_len) grid for the chosen fields."""
    key_lens = list(results)
    lengths = list(results[key_lens[0]])
    present = [f for f in fields if f in results[key_lens[0]][lengths[0]]]
    if title:
        print(title)
    width = 9 * len(present)
    head = f"{'KEY_LEN':8}"
    for L in lengths:
        head += f" | {('--' + str(L) + ' chars--'):^{width}}"
    print(head)
    sub = f"{'':8}"
    for _ in lengths:
        sub += " | " + "".join(f"{_SHORT.get(f, f):<9}" for f in present)
    print(sub)
    print("-" * len(head))
    for kl in key_lens:
        row = f"k={kl:<6}"
        for L in lengths:
            c = results[kl][L]
            row += " | " + "".join(f"{c[f]:<9.2f}" for f in present)
        print(row)


def _shared_cells(neural, classical):
    for k in neural:
        if k not in classical:
            continue
        for L in neural[k]:
            if L in classical[k]:
                yield k, L, neural[k][L], classical[k][L]


def print_comparison(neural, classical,
                     title="PATE against the classical pipeline "
                           "(each solver estimates the period itself)"):
    """The primary comparison: both solvers end to end.

    Period accuracy is reported for each solver separately, so a difference in
    character accuracy can be read against how often each recovered the period.
    EXKEY% is the full key -- period and every shift -- which is the objective
    cryptanalysis is actually measured against.
    """
    print(title)
    print(f"{'d':>4} {'L':>5} | {'CHAR%':>7} {'KEY%':>7} {'EXKEY%':>7} "
          f"| {'CHAR%':>7} {'KEY%':>7} {'EXKEY%':>7} | {'dCHAR':>7} {'dEXKEY':>7}")
    print(f"{'':>4} {'':>5} | {'PATE':^23} | {'classical':^23} |")
    print("-" * 94)
    for k, L, n, c in _shared_cells(neural, classical):
        dn = n.get("exact_key")
        dc = c.get("exact_key")
        ek = (f"{dn:7.1f}", f"{dc:7.1f}", f"{dn - dc:+7.1f}") \
            if (dn is not None and dc is not None) else ("      -", "      -", "      -")
        print(f"{k:>4} {L:>5} | {n['char_acc']:7.1f} {n['key_acc']:7.1f} {ek[0]} "
              f"| {c['char_acc']:7.1f} {c['key_acc']:7.1f} {ek[1]} "
              f"| {n['char_acc'] - c['char_acc']:+7.1f} {ek[2]}")


def print_final_table(neural, blind, ood, classical, nofb=None):
    """PATE (blind, pipeline, out-of-distribution) beside the classical solver.

    BLND%  : period from the first pass alone, no fallback
    KEY%   : period recovered by the full pipeline
    CHAR%  : character accuracy of the pipeline on the validation corpus
    EXKEY% : full key recovered -- period and every shift
    CHAR% (OOD) : pipeline character accuracy on the out-of-distribution corpus
    FB%    : share of inputs sent to the exhaustive fallback
    CHAR% (noFB) : character accuracy with the fallback disabled

    `blind`, `ood` and `nofb` may be None, in which case their column prints "-".
    """
    def f(v, w):
        return f"{v:>{w}.1f}" if v is not None else f"{'-':>{w}}"

    print(f"{'':16}|{'-' * 31} PATE {'-' * 31}|  |------ classical ------|")
    print("  d      L      | BLND%   KEY%   CHAR%   EXKEY%   CHAR% (OOD)   "
          "FB%   CHAR% (noFB)   |  | KEY%   CHAR%   EXKEY% |")
    print("-" * 16 + "+" + "-" * 68 + "  +" + "-" * 24)
    for k, L, n, c in _shared_cells(neural, classical):
        b = blind[k][L] if blind else None
        o = ood[k][L]["char_acc"] if ood else None
        nf = nofb[k][L] if nofb else None
        print(f"{k:>3}{L:>7}      | "
              f"{f(b, 5)}   {f(n['key_acc'], 4)}   {f(n['char_acc'], 5)}   "
              f"{f(n.get('exact_key'), 6)}   {f(o, 11)}   "
              f"{f(n.get('fb_rate'), 3)}   {f(nf, 12)}   |  "
              f"| {f(c['key_acc'], 4)}   {f(c['char_acc'], 5)}   "
              f"{f(c.get('exact_key'), 6)} |")


def print_oracle_comparison(neural, classical,
                            title="secondary condition: the true period is "
                                  "supplied to both solvers"):
    """The oracle condition, reported apart from the primary comparison.

    With period estimation removed from both sides, what remains is shift
    recovery, so the difference measures the decryption mechanism alone.
    """
    print(title)
    print(f"{'d':>4} {'L':>5} | {'CHAR%':>7} {'EXKEY%':>7} "
          f"| {'CHAR%':>7} {'EXKEY%':>7} | {'dCHAR':>7} {'dEXKEY':>7}")
    print(f"{'':>4} {'':>5} | {'PATE':^15} | {'classical':^15} |")
    print("-" * 78)
    for k, L, n, c in _shared_cells(neural, classical):
        no, co = n.get("oracle_char_acc"), c.get("oracle_char_acc")
        if no is None or co is None:
            continue
        ne, ce = n.get("oracle_exact_key"), c.get("oracle_exact_key")
        ek = (f"{ne:7.1f}", f"{ce:7.1f}", f"{ne - ce:+7.1f}") \
            if (ne is not None and ce is not None) else ("      -", "      -", "      -")
        print(f"{k:>4} {L:>5} | {no:7.1f} {ek[0]} | {co:7.1f} {ek[1]} "
              f"| {no - co:+7.1f} {ek[2]}")


def coset_density_table(neural, classical, field="oracle_char_acc"):
    """Accuracy against characters per coset, grouped by L/d.

    Cells that share a coset density come from different (d, L) pairs, so a
    small spread within a group is evidence that the density alone governs the
    result.
    """
    groups = {}
    for k, L, n, c in _shared_cells(neural, classical):
        if field not in n or field not in c:
            continue
        groups.setdefault(round(L / k, 1), []).append((k, L, n[field], c[field]))

    print(f"{field} against coset density")
    print(f"{'L/d':>7} | {'PATE':>7} {'spread':>7} | {'CLS':>7} {'spread':>7} | cells")
    print("-" * 72)
    for rho in sorted(groups):
        g = groups[rho]
        npv = [x[2] for x in g]
        cpv = [x[3] for x in g]
        cells = " ".join(f"(d={k},L={L})" for k, L, _, _ in g)
        print(f"{rho:7.1f} | {sum(npv)/len(npv):7.2f} {max(npv)-min(npv):7.2f} "
              f"| {sum(cpv)/len(cpv):7.2f} {max(cpv)-min(cpv):7.2f} | {cells}")


def density_for_accuracy(results, targets=(80, 90, 95, 98, 99),
                         field="oracle_char_acc"):
    """Characters per coset needed to reach each target accuracy.

    Reads the accuracy curve as a function of L/d and inverts it, which states
    the result as a sample-efficiency requirement rather than a point
    difference on one grid.
    """
    pts = sorted((L / k, cell[field])
                 for k, cols in results.items()
                 for L, cell in cols.items() if field in cell)
    if not pts:
        return {}
    xs = [p[0] for p in pts]
    ys = []
    running = float("-inf")
    for _, y in pts:                    # monotone envelope: the curve is noisy
        running = max(running, y)
        ys.append(running)

    out = {}
    for t in targets:
        if ys[-1] < t:
            out[t] = None
            continue
        for i, y in enumerate(ys):
            if y >= t:
                if i == 0 or ys[i] == ys[i - 1]:
                    out[t] = xs[i]
                else:
                    f = (t - ys[i - 1]) / (ys[i] - ys[i - 1])
                    out[t] = xs[i - 1] + f * (xs[i] - xs[i - 1])
                break
    return out


def print_sample_efficiency(neural, classical, targets=(80, 90, 95, 98, 99),
                            field="oracle_char_acc"):
    """Side-by-side sample-efficiency requirements and their ratio."""
    a = density_for_accuracy(neural, targets, field)
    b = density_for_accuracy(classical, targets, field)
    print(f"characters per coset required to reach a target accuracy ({field})")
    print(f"{'target':>8} | {'PATE':>8} | {'classical':>10} | {'ratio':>7}")
    print("-" * 44)
    for t in targets:
        x, y = a.get(t), b.get(t)
        if x is None or y is None:
            print(f"{t:7d}% | {'--' if x is None else f'{x:8.1f}'} | "
                  f"{'out of range':>10} | {'':>7}")
        else:
            print(f"{t:7d}% | {x:8.1f} | {y:10.1f} | {y/x:6.1f}x")


def print_ood_summary(res):
    """Print the frequency distance and the aggregate figures from run_ood_eval."""
    print(f"OOD corpus        : {res['ood_chars']:>9,} characters")
    print(f"validation corpus : {res['validation_chars']:>9,} characters")
    print(f"character-frequency L1 distance: {res['freq_distance']:.4f}")
    if res["freq_distance"] < 0.05:
        print("  the two corpora have similar distributions; the comparison is weak")

    grid = res["grid"]
    cells = [grid[k][L] for k in grid for L in grid[k]]
    print(f"\nmean character accuracy: {sum(c['char_acc'] for c in cells) / len(cells):.1f}%")
    print(f"mean key-length accuracy: {sum(c['key_acc'] for c in cells) / len(cells):.1f}%")

    if "blind_period" in res:
        bp = res["blind_period"]
        print(f"mean blind period accuracy: {sum(bp.values()) / len(bp):.1f}%")
