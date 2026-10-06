"""Vigenere evaluation harness for the neural models and the classical solver.

Both solvers are measured under the same two conditions, so that period
recovery and shift recovery can be read apart:

  pipeline : the period is estimated from the ciphertext, as in deployment
  oracle   : the true period is supplied, isolating shift recovery

`run_neural_eval`     : per-(key_len x text_len) grid for a Vigenere model --
                        pipeline and oracle character accuracy, key-length
                        accuracy, key recovery and timing.
`run_classical_eval`  : the same grid for the classical Kasiski + coincidence +
                        chi-square solver.
`blind_period_acc`    : per-key-length accuracy of the unconditioned first-pass
                        estimate, for models that have a key-length head.
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
    pos = {c: i for i, c in enumerate(alphabet_chars)}
    n_alpha = len(alphabet_chars)
    votes = [Counter() for _ in range(key_len)]
    for i in range(min(len(cipher), len(plain))):
        c, p = cipher[i], plain[i]
        if c in pos and p in pos:
            votes[i % key_len][(pos[c] - pos[p]) % n_alpha] += 1
    return [v.most_common(1)[0][0] if v else 0 for v in votes]


def apply_key_shifts(cipher, shifts, alphabet_chars):
    """Decrypt a ciphertext under an explicit shift vector."""
    pos = {c: i for i, c in enumerate(alphabet_chars)}
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
    pos = {c: i for i, c in enumerate(alphabet_chars)}
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
def _oracle_decode(model, src_batch, key_len, device, batch_size=64):
    """Second pass only: decrypt with the true period supplied.

    The counterpart of the classical oracle condition. The model's own period
    estimate is bypassed, so what remains is its shift recovery.
    """
    preds = []
    for i in range(0, len(src_batch), batch_size):
        chunk = torch.stack(src_batch[i:i + batch_size])
        kl = torch.full((chunk.size(0),), key_len, dtype=torch.long, device=device)
        plain_logits, _, _ = model(chunk, kl)
        preds.append(plain_logits.argmax(dim=-1).cpu())
    return torch.cat(preds)


def run_neural_eval(model, validation_text, test_lengths, test_key_lens,
                    alphabet_chars, num_samples=500, device="cpu",
                    oracle=True, key_metrics=True):
    """Per-cell accuracy, key recovery and timing for a Vigenere model.

    Reports per cell:
      char_acc            full pipeline, period estimated by the model
      key_acc             period recovered exactly
      oracle_char_acc     true period supplied, isolating shift recovery
      exact_key           full key recovered (period and every shift)
      key_sym_acc         per-symbol key accuracy
      consistent_char_acc decryption re-derived from the coset-majority key
      avg_ms              mean wall-clock latency of the full pipeline
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
            total_ms = 0.0
            srcs, ciphers, plains = [], [], []

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

            cell = {
                "char_acc": correct_chars / max(total_chars, 1) * 100,
                "key_acc": correct_key_len / num_samples * 100,
                "avg_ms": total_ms / num_samples,
            }

            if key_metrics:
                cell["exact_key"] = exact_keys / num_samples * 100
                cell["key_sym_acc"] = key_sym_hits / max(key_sym_total, 1) * 100
                cell["consistent_char_acc"] = cons_correct / max(cons_total, 1) * 100

            if oracle:
                pred = _oracle_decode(model, srcs, key_len, device)
                o_correct = o_total = 0
                for row, ref in zip(pred, plains):
                    dec = "".join(idx2char.get(idx, "") for idx in row.tolist())
                    m = min(len(ref), len(dec))
                    o_correct += sum(a == b for a, b in zip(ref[:m], dec[:m]))
                    o_total += m
                cell["oracle_char_acc"] = o_correct / max(o_total, 1) * 100

            results[key_len][chunk_len] = cell
    return results


# --------------------------------------------------------------------------- #
# classical grid
# --------------------------------------------------------------------------- #

def run_classical_eval(validation_text, test_lengths, test_key_lens,
                       alphabet_chars, num_samples=500, min_k=3, max_k=32,
                       tol=vcl.FUNDAMENTAL_TOL):
    """Classical Vigenere baseline over the same grid and the same conditions.

    `min_k` and `max_k` bound the period search and must cover the experimental
    key-length range; `searched_max_k` records the bound actually reached, which
    is lower than max_k on texts too short to leave two characters per coset.
    """
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    full_clean = _clean(validation_text, char2idx)
    pos = {c: i for i, c in enumerate(alphabet_chars)}

    results = {}
    for key_len in test_key_lens:
        results[key_len] = {}
        for chunk_len in test_lengths:
            total_chars = correct_chars = correct_key_len = 0
            oracle_total = oracle_correct = 0
            exact_key = key_sym_hits = key_sym_total = 0
            total_ms = 0.0
            searched_max_k = 0

            for i in range(num_samples):
                start = (i * chunk_len * 13) % max(1, len(full_clean) - chunk_len)
                plain = full_clean[start:start + chunk_len]
                key = get_fixed_key(key_len, i, alphabet_chars)
                cipher, clean_plain = vinf.encrypt_vigenere(plain, key, char2idx, idx2char)
                searched_max_k = vcl.searchable_range(cipher, min_k, max_k)[1]

                # --- full pipeline ---
                t0 = time.perf_counter()
                decoded, pred_k, _ = vcl.classical_vigenere_solve(
                    cipher, min_k, max_k, tol=tol)
                total_ms += (time.perf_counter() - t0) * 1000

                m = min(len(clean_plain), len(decoded))
                correct_chars += sum(a == b for a, b in zip(clean_plain[:m], decoded[:m]))
                total_chars += m
                correct_key_len += int(pred_k == key_len)

                # --- oracle: true period supplied ---
                o_decoded, _, o_shifts = vcl.classical_vigenere_solve(
                    cipher, min_k, max_k, true_key_len=key_len, tol=tol)
                om = min(len(clean_plain), len(o_decoded))
                oracle_correct += sum(a == b for a, b in zip(clean_plain[:om], o_decoded[:om]))
                oracle_total += om

                # --- key recovery under the oracle period ---
                true_shifts = [pos[c] for c in key]
                hits = sum(a == b for a, b in zip(o_shifts, true_shifts))
                key_sym_hits += hits
                key_sym_total += len(true_shifts)
                exact_key += int(hits == len(true_shifts))

            results[key_len][chunk_len] = {
                "char_acc":        correct_chars / max(total_chars, 1) * 100,
                "key_acc":         correct_key_len / num_samples * 100,
                "oracle_char_acc": oracle_correct / max(oracle_total, 1) * 100,
                "exact_key":       exact_key / num_samples * 100,
                "key_sym_acc":     key_sym_hits / max(key_sym_total, 1) * 100,
                "avg_ms":          total_ms / num_samples,
                "searched_max_k":  searched_max_k,
            }
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
    short = {"char_acc": "CHAR%", "key_acc": "KEY%", "oracle_char_acc": "ORAC%",
             "exact_key": "EXKEY%", "key_sym_acc": "KSYM%",
             "consistent_char_acc": "CONS%", "avg_ms": "ms"}
    for _ in lengths:
        sub += " | " + "".join(f"{short.get(f, f):<9}" for f in present)
    print(sub)
    print("-" * len(head))
    for kl in key_lens:
        row = f"k={kl:<6}"
        for L in lengths:
            c = results[kl][L]
            row += " | " + "".join(f"{c[f]:<9.2f}" for f in present)
        print(row)


def print_comparison(neural, classical, title="pipeline and oracle, side by side"):
    """PATE against the classical solver under both conditions.

    The pipeline columns are the fair comparison: each solver estimates the
    period itself. The oracle columns remove period estimation from both sides
    and leave only shift recovery.
    """
    print(title)
    print(f"{'d':>4} {'L':>5} | {'PATE':>7} {'CLS':>7} {'diff':>7} "
          f"| {'PATE-o':>7} {'CLS-o':>7} {'diff':>7}")
    print("-" * 66)
    for k in neural:
        if k not in classical:
            continue
        for L in neural[k]:
            if L not in classical[k]:
                continue
            n, c = neural[k][L], classical[k][L]
            no = n.get("oracle_char_acc")
            co = c.get("oracle_char_acc")
            left = f"{n['char_acc']:7.1f} {c['char_acc']:7.1f} {n['char_acc']-c['char_acc']:+7.1f}"
            right = ("  " + "-" * 5) * 3 if (no is None or co is None) \
                else f"{no:7.1f} {co:7.1f} {no-co:+7.1f}"
            print(f"{k:>4} {L:>5} | {left} | {right}")


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
