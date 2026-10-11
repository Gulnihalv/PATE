"""Evaluation harness for the substitution models and classical solvers.

`run_model_comparison` : raw/consistent accuracy, timing and key-level metrics for N models.
`run_classical_eval`   : the same measures for Hill Climbing / Simulated Annealing.

Both harnesses now share the identical preprocessing, plaintext slices and
encryption keys, so model and baseline solve the same problem instances:

  * text is reduced to the model's alphabet (space-free) in BOTH paths;
  * slices are anchored, so a short sample is a prefix of a longer one and the
    length curve is not confounded with corpus region;
  * key i is the same for the model and for the classical solver.

Per-sample accuracies are returned alongside the means so that paired
bootstrap intervals can be computed without re-running inference.
"""

import time
import random

from models.substitution_common import inference as infer
from models.substitution_common import classical as cl


# --------------------------------------------------------------------------- #
#  Shared helpers                                                              #
# --------------------------------------------------------------------------- #

def get_fixed_key(seed_val: int, alphabet_chars: str) -> dict:
    """Deterministic plaintext->ciphertext key. Uses a local RNG."""
    rng = random.Random(seed_val)
    real = list(alphabet_chars)
    shuffled = list(alphabet_chars)
    rng.shuffle(shuffled)
    return {s: t for s, t in zip(real, shuffled)}


def _slice(full_text, i, chunk_len, anchor):
    """Anchored slice: the start position depends on `anchor`, not on chunk_len,
    so the L=30 sample is a prefix of the L=1000 sample drawn with the same i."""
    start = (i * anchor * 13) % max(1, len(full_text) - anchor)
    return full_text[start:start + chunk_len]


def _key_metrics(cipher, decoded, key_map):
    """Key-level recovery from a decryption.

    Returns (n_correct, n_observed): how many of the distinct cipher symbols
    present in this ciphertext were mapped to the right plaintext symbol.
    Symbols that never occur cannot be recovered, so the denominator is the
    number observed, and coverage is reported separately.
    """
    true_dec = {c: p for p, c in key_map.items()}      # cipher -> plain
    pred_dec = {}
    for c, p in zip(cipher, decoded):
        pred_dec.setdefault(c, p)                       # CONSISTENT is bijective
    n_obs = len(pred_dec)
    n_ok = sum(1 for c, p in pred_dec.items() if true_dec.get(c) == p)
    return n_ok, n_obs


def _sync(device):
    if str(device).startswith("cuda"):
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  Neural models                                                               #
# --------------------------------------------------------------------------- #

def run_model_comparison(models, validation_text, test_lengths,
                         alphabet_chars, num_samples=500, seq_len=512,
                         stride=400, device="cpu", warmup=5):
    """Compare N models.

    models : {"BTE": model_bte, "FATE": model_fate}
    Returns results[name][chunk_len] = {
        "raw", "cons"              : mean character accuracy (%)
        "raw_per_sample", "cons_per_sample" : lists of length num_samples (0-1)
        "raw_ms", "cons_ms"        : mean latency per sample (ms)
        "key_acc", "key_cov"       : mean % of observed cipher symbols mapped
                                     correctly under CONSISTENT, and mean
                                     fraction of the alphabet observed
        "exact_key"                : % of samples with every observed symbol right
    }
    """
    char2idx, idx2char = infer.build_vocab(alphabet_chars)
    text = infer.clean_text(validation_text, alphabet_chars)
    anchor = max(test_lengths)
    n_alpha = len(alphabet_chars)

    # warm up CUDA kernels / autograd caches so the first timed call is not an outlier
    if warmup:
        probe = text[:seq_len]
        key0 = get_fixed_key(0, alphabet_chars)
        probe_c = probe.translate(str.maketrans(key0))
        for model in models.values():
            for _ in range(warmup):
                infer.infer_raw(model, probe_c, char2idx, idx2char, seq_len, stride, device)
        _sync(device)

    results = {name: {} for name in models}

    for chunk_len in test_lengths:
        acc = {name: dict(raw=[], cons=[], raw_ms=0.0, cons_ms=0.0,
                          k_ok=0, k_obs=0, exact=0) for name in models}

        for i in range(num_samples):
            key_map = get_fixed_key(i, alphabet_chars)
            plain = _slice(text, i, chunk_len, anchor)
            cipher = plain.translate(str.maketrans(key_map))
            m = len(plain)
            if m == 0:
                continue

            for name, model in models.items():
                a = acc[name]

                _sync(device); t0 = time.perf_counter()
                raw = infer.infer_raw(model, cipher, char2idx, idx2char,
                                      seq_len, stride, device)
                _sync(device); a["raw_ms"] += (time.perf_counter() - t0) * 1000
                a["raw"].append(sum(x == y for x, y in zip(plain, raw[:m])) / m)

                _sync(device); t0 = time.perf_counter()
                cons = infer.infer_consistent(model, cipher, char2idx, idx2char,
                                              seq_len, stride, device)
                _sync(device); a["cons_ms"] += (time.perf_counter() - t0) * 1000
                a["cons"].append(sum(x == y for x, y in zip(plain, cons[:m])) / m)

                n_ok, n_obs = _key_metrics(cipher, cons[:m], key_map)
                a["k_ok"] += n_ok
                a["k_obs"] += n_obs
                a["exact"] += int(n_ok == n_obs and n_obs > 0)

        for name in models:
            a = acc[name]
            n = max(len(a["raw"]), 1)
            results[name][chunk_len] = {
                "raw":  sum(a["raw"]) / n * 100,
                "cons": sum(a["cons"]) / n * 100,
                "raw_per_sample":  a["raw"],
                "cons_per_sample": a["cons"],
                "raw_ms":  a["raw_ms"] / n,
                "cons_ms": a["cons_ms"] / n,
                "key_acc": a["k_ok"] / max(a["k_obs"], 1) * 100,
                "key_cov": a["k_obs"] / n / n_alpha * 100,
                "exact_key": a["exact"] / n * 100,
            }

    return results


# --------------------------------------------------------------------------- #
#  Classical solvers                                                           #
# --------------------------------------------------------------------------- #

def build_language_model(lm_text, alphabet_chars, w_static=0.6):
    """Blend the hand-entered Turkish tables with tables estimated from `lm_text`.

    NOTE on the blend: the static tables in classical.py were compiled from
    spaced Turkish, so in space-free text every n-gram that straddles a word
    boundary is absent from them and falls to the floor. Set w_static=0.0 to
    use corpus-estimated tables only, which match the preprocessing exactly.
    """
    dg_s, floor_dg_s, tg_s, floor_tg_s = cl.build_static_tables()
    bg_c, floor_bg_c, tg_c, floor_tg_c = cl.build_corpus_tables(
        infer.clean_text(lm_text, alphabet_chars))

    w_c = 1.0 - w_static

    # bigrams with bigrams, trigrams with trigrams
    keys_b = set(dg_s) | set(bg_c)
    bg_lp = {k: w_static * dg_s.get(k, floor_dg_s) + w_c * bg_c.get(k, floor_bg_c)
             for k in keys_b}
    floor_b = w_static * floor_dg_s + w_c * floor_bg_c

    keys_t = set(tg_s) | set(tg_c)
    tg_lp = {k: w_static * tg_s.get(k, floor_tg_s) + w_c * tg_c.get(k, floor_tg_c)
             for k in keys_t}
    floor_t = w_static * floor_tg_s + w_c * floor_tg_c

    return tg_lp, floor_t, bg_lp, floor_b


def run_classical_eval(method, validation_text, test_lengths, alphabet_chars,
                       num_samples=500, lm_text=None, w_static=0.6,
                       **solver_kwargs):
    """method: 'hill_climbing' | 'simulated_annealing'.

    `lm_text` is the corpus the solver's language model is estimated from.
    It defaults to `validation_text`, which gives the baseline a language model
    fitted to the very text it decrypts; pass a held-out corpus to remove that
    advantage. Returns the same measures as run_model_comparison.

    Timing is single-threaded CPU and is NOT comparable to GPU model latency.
    """
    text = infer.clean_text(validation_text, alphabet_chars)
    lm_src = validation_text if lm_text is None else lm_text
    anchor = max(test_lengths)
    n_alpha = len(alphabet_chars)

    if method == "hill_climbing":
        bigram_table, floor = cl.build_bigram_table(
            infer.clean_text(lm_src, alphabet_chars))
    elif method == "simulated_annealing":
        tg_lp, floor_t, bg_lp, floor_b = build_language_model(
            lm_src, alphabet_chars, w_static=w_static)
    else:
        raise ValueError(method)

    results = {}
    for chunk_len in test_lengths:
        per_sample, times = [], []
        k_ok = k_obs = exact = 0

        for i in range(num_samples):
            key_map = get_fixed_key(i, alphabet_chars)      # same key as the models
            plain = _slice(text, i, chunk_len, anchor)
            if not plain:
                continue
            cipher = plain.translate(str.maketrans(key_map))

            t0 = time.perf_counter()
            if method == "hill_climbing":
                decoded, _ = cl.hill_climbing_solve(
                    cipher, bigram_table, floor, seed=i, **solver_kwargs)
            else:
                decoded, _ = cl.simulated_annealing_solve(
                    cipher, tg_lp, floor_t, bg_lp, floor_b,
                    rng_seed=i, **solver_kwargs)
            times.append((time.perf_counter() - t0) * 1000)

            m = min(len(plain), len(decoded))
            per_sample.append(sum(a == b for a, b in zip(plain[:m], decoded[:m])) / m)

            n_ok, n_obs = _key_metrics(cipher, decoded[:m], key_map)
            k_ok += n_ok
            k_obs += n_obs
            exact += int(n_ok == n_obs and n_obs > 0)

        n = max(len(per_sample), 1)
        results[chunk_len] = {
            "acc": sum(per_sample) / n * 100,
            "per_sample": per_sample,
            "avg_ms": sum(times) / max(len(times), 1),
            "key_acc": k_ok / max(k_obs, 1) * 100,
            "key_cov": k_obs / n / n_alpha * 100,
            "exact_key": exact / n * 100,
        }
    return results


# --------------------------------------------------------------------------- #
#  Paired bootstrap                                                            #
# --------------------------------------------------------------------------- #

def paired_bootstrap(a_per_sample, b_per_sample, n_boot=10000, seed=0):
    """95% CI for mean(a - b) over matched samples. Resamples at the TEXT level,
    not the character level: characters inside one sample share a key and are
    not independent."""
    import numpy as np
    a = np.asarray(a_per_sample, dtype=float)
    b = np.asarray(b_per_sample, dtype=float)
    assert len(a) == len(b) and len(a) > 0, "unpaired or empty inputs"
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return d.mean() * 100, lo * 100, hi * 100


# --------------------------------------------------------------------------- #
#  Printing                                                                    #
# --------------------------------------------------------------------------- #

def print_comparison_table(results):
    names = list(results)
    lengths = sorted(results[names[0]])
    head = f"{'LEN':<6}"
    for n in names:
        head += f" | {n+' RAW%':<8} {n+' CONS%':<9} {n+' ms':<7} {n+' KEY%':<8}"
    print(head)
    print("-" * len(head))
    for L in lengths:
        row = f"{L:<6}"
        for n in names:
            r = results[n][L]
            row += (f" | {r['raw']:<8.2f} {r['cons']:<9.2f} "
                    f"{r['cons_ms']:<7.2f} {r['key_acc']:<8.2f}")
        print(row)


def print_paired_table(results, a="FATE", b="BTE", mode="cons"):
    lengths = sorted(results[a])
    print(f"{'LEN':<6} | {a+' %':<8} {b+' %':<8} {'DIFF':<8} {'95% CI'}")
    print("-" * 56)
    for L in lengths:
        m, lo, hi = paired_bootstrap(results[a][L][f"{mode}_per_sample"],
                                     results[b][L][f"{mode}_per_sample"])
        print(f"{L:<6} | {results[a][L][mode]:<8.2f} {results[b][L][mode]:<8.2f} "
              f"{m:<+8.2f} [{lo:+.2f}, {hi:+.2f}]")