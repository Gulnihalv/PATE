"""Vigenere evaluation harness for the neural models and the classical solver.

`run_neural_eval`     : per-(key_len x text_len) character accuracy, key-length
                        accuracy and timing for a Vigenere model.
`run_classical_eval`  : the same grid for the classical IoC+Kasiski+frequency solver.
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


def run_neural_eval(model, validation_text, test_lengths, test_key_lens,
                    alphabet_chars, num_samples=500, device="cpu"):
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    full_clean = _clean(validation_text, char2idx)

    results = {}
    for key_len in test_key_lens:
        results[key_len] = {}
        for chunk_len in test_lengths:
            total_chars = correct_chars = correct_key_len = 0
            total_ms = 0.0
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
                correct_key_len += int(result["best_key_len"] == key_len)

            results[key_len][chunk_len] = {
                "char_acc": correct_chars / max(total_chars, 1) * 100,
                "key_acc": correct_key_len / num_samples * 100,
                "avg_ms": total_ms / num_samples,
            }
    return results


def run_classical_eval(validation_text, test_lengths, test_key_lens,
                       alphabet_chars, num_samples=500, min_k=3, max_k=32):
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    full_clean = _clean(validation_text, char2idx)

    results = {}
    for key_len in test_key_lens:
        results[key_len] = {}
        for chunk_len in test_lengths:
            total_chars = correct_chars = correct_key_len = 0
            total_ms = 0.0
            for i in range(num_samples):
                start = (i * chunk_len * 13) % max(1, len(full_clean) - chunk_len)
                plain = full_clean[start:start + chunk_len]
                key = get_fixed_key(key_len, i, alphabet_chars)
                cipher, clean_plain = vinf.encrypt_vigenere(plain, key, char2idx, idx2char)

                t0 = time.perf_counter()
                decoded, pred_k = vcl.classical_vigenere_solve(cipher, min_k, max_k)
                total_ms += (time.perf_counter() - t0) * 1000

                m = min(len(clean_plain), len(decoded))
                correct_chars += sum(a == b for a, b in zip(clean_plain[:m], decoded[:m]))
                total_chars += m
                correct_key_len += int(pred_k == key_len)

            results[key_len][chunk_len] = {
                "char_acc": correct_chars / max(total_chars, 1) * 100,
                "key_acc": correct_key_len / num_samples * 100,
                "avg_ms": total_ms / num_samples,
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


def print_grid(results, title=""):
    """Print the (key_len x text_len) grid with CHAR% / KEY% / ms."""
    key_lens = list(results)
    lengths = list(results[key_lens[0]])
    if title:
        print(title)
    head = f"{'KEY_LEN':8}"
    for L in lengths:
        head += f" | {('--' + str(L) + ' chars--'):^26}"
    print(head)
    sub = f"{'':8}"
    for _ in lengths:
        sub += f" | {'CHAR%':<8} {'KEY%':<8} {'ms':<6}"
    print(sub)
    print("-" * len(head))
    for kl in key_lens:
        row = f"k={kl:<6}"
        for L in lengths:
            c = results[kl][L]
            row += f" | {c['char_acc']:<8.2f} {c['key_acc']:<8.2f} {c['avg_ms']:<6.2f}"
        print(row)


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
