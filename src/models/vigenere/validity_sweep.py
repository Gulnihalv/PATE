"""Calibration of the verification threshold.

The decoder accepts its blind period estimate when the validity head scores it
above a threshold and otherwise falls back to scoring every candidate period.
The threshold is therefore a free parameter, and the paper has to say what it
is and how it was chosen.

Every sample is decoded once under its blind estimate and once under the
exhaustive pass, and its validity score is recorded; the outcome at any
threshold then follows without running the model again, since a threshold only
decides which of the two already-computed decodings is kept.
"""

import collections

import torch

from models.vigenere import inference as vinf
from models.vigenere.evaluation import get_fixed_key, _clean


@torch.no_grad()
def collect(model, text, alphabet_chars, test_lengths=(128, 256, 512),
            test_key_lens=None, num_samples=200, device="cpu",
            unknown_key=0, batch_size=64):
    """Per-sample validity score and the accuracy of both decodings."""
    char2idx, idx2char = vinf.build_vocab(alphabet_chars)
    clean = _clean(text, char2idx)
    if test_key_lens is None:
        test_key_lens = list(range(model.min_key_len, model.max_key_len + 1))
    cands = list(range(model.min_key_len, model.max_key_len + 1))

    out = []
    for key_len in test_key_lens:
        for chunk_len in test_lengths:
            srcs, refs = [], []
            for i in range(num_samples):
                start = (i * chunk_len * 13) % max(1, len(clean) - chunk_len)
                key = get_fixed_key(key_len, i, alphabet_chars)
                cipher, ref = vinf.encrypt_vigenere(clean[start:start + chunk_len],
                                                    key, char2idx, idx2char)
                srcs.append(vinf.text_to_tensor(cipher, char2idx, device))
                refs.append(ref)

            for s in range(0, len(srcs), batch_size):
                src = torch.stack(srcs[s:s + batch_size])
                ref = refs[s:s + batch_size]
                B = src.size(0)

                # blind pass: the model's own period estimate
                unk = torch.full((B,), unknown_key, dtype=torch.long, device=device)
                _, _, kl_logits = model(src, unk)
                k_hat = kl_logits.argmax(-1) + model.min_key_len

                # every candidate period, scored by the validity head
                best_v = torch.full((B,), -1e9, device=device)
                best_pred = None
                v_at_khat = torch.zeros(B, device=device)
                for k in cands:
                    kk = torch.full((B,), k, dtype=torch.long, device=device)
                    logits, val, _ = model(src, kk)
                    v = torch.sigmoid(val.reshape(B, -1).mean(-1))   # <- adjust if the head differs
                    pred = logits.argmax(-1)
                    if best_pred is None:
                        best_pred = pred.clone()
                    take = v > best_v
                    best_v = torch.where(take, v, best_v)
                    best_pred[take] = pred[take]
                    v_at_khat = torch.where(k_hat == k, v, v_at_khat)

                # the decoding the blind estimate would have produced
                blind_logits, _, _ = model(src, k_hat)
                blind_pred = blind_logits.argmax(-1)

                for b in range(B):
                    def acc(row):
                        dec = "".join(idx2char.get(i, "") for i in row.tolist())
                        m = min(len(ref[b]), len(dec))
                        return sum(x == y for x, y in zip(ref[b][:m], dec[:m])) / max(m, 1)
                    out.append({
                        "key_len": key_len, "text_len": chunk_len,
                        "validity": float(v_at_khat[b]),
                        "acc_blind": acc(blind_pred[b]),
                        "acc_fallback": acc(best_pred[b]),
                    })
    return out


def sweep(records, thresholds=None, tolerance=0.5):
    """Character accuracy and fallback rate at each threshold.

    Picks the smallest threshold whose accuracy is within `tolerance` points of
    the best threshold on the grid -- the cheapest setting that gives up
    essentially none of what verification is worth. Accuracy is not monotone in
    the threshold: the exhaustive pass can replace a correct blind decoding with
    a wrong one, so always verifying is not the ceiling.

    The validity scores crowd near 1, so the default grid is dense there.
    """
    if thresholds is None:
        thresholds = [0.0, 0.5, 0.9, 0.95, 0.97, 0.98, 0.99,
                      0.995, 0.999, 0.9995, 0.9999, 1.0]
    n = len(records)
    ceiling = sum(max(r["acc_blind"], r["acc_fallback"]) for r in records) / n * 100

    rows = []
    for t in thresholds:
        acc = fb = 0.0
        for r in records:
            use_fb = r["validity"] < t
            acc += r["acc_fallback"] if use_fb else r["acc_blind"]
            fb += use_fb
        rows.append((t, acc / n * 100, fb / n * 100))

    always = rows[-1][1]
    best_t, best, _ = max(rows, key=lambda r: r[1])
    chosen = next(t for t, a, _ in rows if a >= best - tolerance)

    print(f"{'threshold':>10} | {'CHAR%':>7} | {'fallback%':>10}")
    print("-" * 34)
    for t, a, f in rows:
        mark = "  <- chosen" if abs(t - chosen) < 1e-9 else ""
        print(f"{t:10g} | {a:7.2f} | {f:10.1f}{mark}")
    print(f"\nalways verifying: {always:.2f}%   never verifying: {rows[0][1]:.2f}%")
    print(f"best on the grid: {best:.2f}% at threshold {best_t:g}")
    print(f"oracle pick of the two: {ceiling:.2f}%")
    print(f"chosen threshold {chosen:g}: within {tolerance} points of the best, "
          f"at {dict((t, f) for t, _, f in rows)[chosen]:.1f}% fallback")
    return rows, chosen


def per_cell(records, t_old=0.5, t_new=0.95):
    """Where the fallback lands, and what moving the threshold does per cell.

    Only samples with t_old <= validity < t_new change decoding between the two
    thresholds; `moved` counts them, and `up` / `down` split them by whether the
    exhaustive pass beat or lost to the blind decoding. The net accuracy change
    rests on these few samples, so the counts matter more than the percentage.
    """
    agg = collections.defaultdict(lambda: dict(n=0, fb_old=0, fb_new=0, sw=0,
                                               up=0, down=0, a_old=0.0, a_new=0.0))
    for r in records:
        c = agg[(r["key_len"], r["text_len"])]
        old, new = r["validity"] < t_old, r["validity"] < t_new
        c["n"] += 1
        c["fb_old"] += old
        c["fb_new"] += new
        c["a_old"] += r["acc_fallback"] if old else r["acc_blind"]
        c["a_new"] += r["acc_fallback"] if new else r["acc_blind"]
        if new and not old:                     # only these samples change
            c["sw"] += 1
            c["up"] += r["acc_fallback"] > r["acc_blind"]
            c["down"] += r["acc_fallback"] < r["acc_blind"]

    print(f"{'d':>4} {'L':>5} | {'FB%':>6} {'FB%':>6} | {'moved':>5} {'up':>4} {'down':>4} "
          f"| {'CHAR%':>7} {'CHAR%':>7} {'diff':>7}")
    print(f"{'':>4} {'':>5} | {f't={t_old:g}':>6} {f't={t_new:g}':>6} | {'':>15} "
          f"| {f't={t_old:g}':>7} {f't={t_new:g}':>7}")
    print("-" * 72)
    for (d, L), c in sorted(agg.items()):
        n = c["n"]
        print(f"{d:>4} {L:>5} | {100*c['fb_old']/n:6.1f} {100*c['fb_new']/n:6.1f} "
              f"| {c['sw']:>5} {c['up']:>4} {c['down']:>4} "
              f"| {100*c['a_old']/n:7.2f} {100*c['a_new']/n:7.2f} "
              f"{100*(c['a_new']-c['a_old'])/n:+7.2f}")
    return dict(agg)
