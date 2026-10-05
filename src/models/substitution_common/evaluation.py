"""Evaluation harness for the substitution models and classical solvers.

`run_model_comparison` : raw/consistent character accuracy + timing for N models.
`run_classical_eval`   : accuracy + timing for Hill Climbing / Simulated Annealing.

Functions return structured dicts so the printed table and the plots read from
the same data. Everything is passed in as parameters (no globals).
"""

import time
import random

from models.substitution_common import inference as infer
from models.substitution_common import classical as cl


def get_fixed_key(seed_val: int, alphabet_chars: str) -> dict:
    """Deterministic encryption key (depends only on the seed)."""
    random.seed(seed_val)
    real = list(alphabet_chars)
    shuffled = list(alphabet_chars)
    random.shuffle(shuffled)
    return {s: t for s, t in zip(real, shuffled)}


def _slice(full_text, i, chunk_len):
    start = (i * chunk_len * 13) % max(1, len(full_text) - chunk_len)
    return full_text[start:start + chunk_len]


def run_model_comparison(models, validation_text, test_lengths,
                         alphabet_chars, num_samples=500, seq_len=256,
                         stride=200, device="cpu"):
    """Compare N models.

    models : {"FATE": model_fate, "BTE": model_bte}  (Lightning module or nn.Module)
    Returns: results[name] = {"len": [...], "raw": [...], "cons": [...],
                              "raw_ms": [...], "cons_ms": [...]}
    """
    char2idx, idx2char = infer.build_vocab(alphabet_chars)
    validation_text = infer.clean_text(validation_text, alphabet_chars)  # model is space-free
    results = {name: {"len": [], "raw": [], "cons": [], "raw_ms": [], "cons_ms": []}
               for name in models}

    for chunk_len in test_lengths:
        agg = {name: dict(raw_c=0, cons_c=0, raw_ms=0.0, cons_ms=0.0) for name in models}
        total_chars = 0

        for i in range(num_samples):
            key_map = get_fixed_key(i, alphabet_chars)
            table = str.maketrans(key_map)
            plain = _slice(validation_text, i, chunk_len)
            cipher = plain.translate(table)
            m = len(plain)
            total_chars += m

            for name, model in models.items():
                t0 = time.perf_counter()
                raw = infer.infer_raw(model, cipher, char2idx, idx2char, seq_len, stride, device)
                agg[name]["raw_ms"] += (time.perf_counter() - t0) * 1000
                agg[name]["raw_c"] += sum(a == b for a, b in zip(plain, raw[:m]))

                t0 = time.perf_counter()
                cons = infer.infer_consistent(model, cipher, char2idx, idx2char, seq_len, stride, device)
                agg[name]["cons_ms"] += (time.perf_counter() - t0) * 1000
                agg[name]["cons_c"] += sum(a == b for a, b in zip(plain, cons[:m]))

        for name in models:
            results[name]["len"].append(chunk_len)
            results[name]["raw"].append(agg[name]["raw_c"] / total_chars * 100)
            results[name]["cons"].append(agg[name]["cons_c"] / total_chars * 100)
            results[name]["raw_ms"].append(agg[name]["raw_ms"] / num_samples)
            results[name]["cons_ms"].append(agg[name]["cons_ms"] / num_samples)

    return results


def run_classical_eval(method, validation_text, test_lengths, alphabet_chars,
                       num_samples=500, **solver_kwargs):
    """method: 'hill_climbing' | 'simulated_annealing'. Returns {len: {acc, avg_ms}}."""
    if method == "hill_climbing":
        bigram_table, floor = cl.build_bigram_table(validation_text)
    elif method == "simulated_annealing":
        # Static (digram/trigram) and corpus tables are blended in this specific
        # pairing. The ordering is intentional and must not be changed, as it
        # alters the scoring.
        dg, fd, tg, ft = cl.build_static_tables()        # (digram_lp, floor_d, trigram_lp, floor_t)
        bg_lp_c, fb_c, tg_lp_c, ft_c = cl.build_corpus_tables(validation_text)

        # Pairing: tg_static <- digram, bg_static <- trigram
        tg_static, floor_ts = dg, fd        # tg_static holds digrams
        bg_static, floor_bs = tg, ft        # bg_static holds trigrams
        tg_corpus, floor_tc = tg_lp_c, ft_c
        bg_corpus, floor_bc = bg_lp_c, fb_c

        all_t = set(tg_static) | set(tg_corpus)
        tg_lp = {k: 0.6 * tg_static.get(k, floor_ts) + 0.4 * tg_corpus.get(k, floor_tc) for k in all_t}
        floor_t = 0.6 * floor_ts + 0.4 * floor_tc
        all_b = set(bg_static) | set(bg_corpus)
        bg_lp = {k: 0.6 * bg_static.get(k, floor_bs) + 0.4 * bg_corpus.get(k, floor_bc) for k in all_b}
        floor_b = 0.6 * floor_bs + 0.4 * floor_bc
    else:
        raise ValueError(method)

    results = {}
    for chunk_len in test_lengths:
        correct, total, times = 0, 0, []
        for i in range(num_samples):
            plain = _slice(validation_text, i, chunk_len)
            plain_clean = "".join(c for c in plain.lower() if c in cl.ALPHA_SET)
            if not plain_clean:
                continue
            key_map = get_fixed_key(chunk_len * 10000 + i, alphabet_chars)
            cipher = plain_clean.translate(str.maketrans(key_map))

            t0 = time.perf_counter()
            if method == "hill_climbing":
                decoded, _ = cl.hill_climbing_solve(
                    cipher, bigram_table, floor, seed=chunk_len * 10000 + i, **solver_kwargs)
            else:
                decoded, _ = cl.simulated_annealing_solve(
                    cipher, tg_lp, floor_t, bg_lp, floor_b,
                    rng_seed=chunk_len * 10000 + i, **solver_kwargs)
            times.append((time.perf_counter() - t0) * 1000)

            ml = min(len(plain_clean), len(decoded))
            correct += sum(a == b for a, b in zip(plain_clean[:ml], decoded[:ml]))
            total += ml

        results[chunk_len] = {
            "acc": correct / max(total, 1) * 100,
            "avg_ms": sum(times) / max(len(times), 1),
        }
    return results


def print_comparison_table(results):
    """Print RAW%, RAW_ms, CONS%, CONS_ms columns per model."""
    names = list(results)
    lengths = results[names[0]]["len"]
    head = f"{'UZUNLUK':<8}"
    for n in names:
        head += f" | {n+'_R%':<7} {n+'_R_ms':<8} {n+'_C%':<7} {n+'_C_ms':<8}"
    print(head)
    print("-" * len(head))
    for k, length in enumerate(lengths):
        row = f"{length:<8}"
        for n in names:
            r = results[n]
            row += (f" | {r['raw'][k]:<7.2f} {r['raw_ms'][k]:<8.2f} "
                    f"{r['cons'][k]:<7.2f} {r['cons_ms'][k]:<8.2f}")
        print(row)
