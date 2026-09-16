"""Classical (learning-free) substitution solvers used as baselines.

  - Hill Climbing      : bigram log-probability score, with random restarts
  - Simulated Annealing: trigram + bigram score, exponential cooling schedule

Turkish digram/trigram frequencies are embedded and optionally blended with
corpus tables. This module is model-independent (standard library only).
"""

import math
import random

ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz "   # 30 characters (space included)
ALPHA_SET = set(ALPHABET)

# Approximate Turkish letter-frequency order (smart SA start)
TR_ORDER = list(" aeinrlktdsmybouğşçhzgüöpvcfjı")

# Turkish digram frequencies (per 100,000)
RAW_DIGRAMS = {
    "ar": 2273, "la": 2013, "an": 1891, "er": 1822, "in": 1674, "le": 1640,
    "de": 1475, "en": 1408, "da": 1311, "ir": 1282, "bi": 1253, "ka": 1155,
    "ya": 1135, "ma": 1044, "di": 1021, "nd": 980, "ra": 976, "al": 974,
    "ak": 967, "il": 870, "ri": 860, "me": 785, "li": 782, "or": 782,
    "ne": 738, "ba": 718, "ni": 716, "el": 710, "ay": 698, "yo": 686,
    "ek": 683, "rd": 681, "ta": 670, "am": 638, "sa": 624, "ki": 618,
    "iy": 619, "un": 606, "na": 602, "ad": 592, "ye": 588, "ol": 586,
    "si": 578, "re": 566, "mi": 564, "te": 562, "et": 560, "im": 541,
    "ti": 537, "ha": 528, "as": 527, "bu": 516, "ve": 508, "nl": 496,
    "em": 494, "du": 487, "ge": 480, "at": 479, "se": 457, "ed": 452,
    "ur": 452, "on": 452, "kl": 447, "is": 434, "be": 433, "ke": 424,
    "ey": 421, "es": 411, "ik": 407, "rl": 393, "ca": 379, "ld": 362,
    "ce": 361, "nu": 359, "iz": 353, "lm": 353, "ru": 349, "gi": 347,
    "az": 343, "ah": 338, "yl": 324,
}

# Turkish trigram frequencies (per 100,000)
RAW_TRIGRAMS = {
    "lar": 1237, "bir": 952, "ler": 949, "eri": 764, "ari": 757, "yor": 643,
    "ara": 521, "nda": 482, "ini": 432, "asi": 387, "den": 383, "nde": 383,
    "rin": 372, "ile": 367, "ani": 362, "ama": 357, "nla": 338, "dan": 338,
    "ind": 336, "edi": 326, "ada": 321, "aya": 316, "kar": 299, "ala": 298,
    "lan": 296, "eni": 294, "sin": 294, "esi": 283, "nin": 280, "yle": 277,
    "adi": 273, "ine": 266, "anl": 263, "ere": 262, "kla": 262, "ali": 258,
    "iye": 255, "bil": 246, "ili": 245, "bas": 243, "ard": 242, "rdu": 231,
    "mis": 229, "ola": 227, "igi": 226, "eme": 223, "egi": 223, "ina": 222,
    "ana": 220, "ken": 218, "ici": 217, "iyo": 217, "rla": 216, "agi": 194,
    "ord": 194, "gel": 194, "man": 192, "aca": 192, "oyl": 191, "kad": 187,
    "erd": 183, "rak": 177, "oru": 178, "ver": 170, "emi": 169, "rdi": 169,
    "son": 168, "ila": 167, "ben": 166, "cak": 165, "iri": 163, "eye": 163,
    "cik": 160, "kan": 159,
}


def apply_key(cipher: str, key: dict) -> str:
    return "".join(key.get(c, c) for c in cipher)


# --- Table builders ---

def build_bigram_table(text: str):
    bg, total = {}, 0
    for i in range(len(text) - 1):
        b = text[i:i + 2]
        if all(c in ALPHABET for c in b):
            bg[b] = bg.get(b, 0) + 1
            total += 1
    floor = math.log(0.01 / max(total, 1))
    return {k: math.log(v / total) for k, v in bg.items()}, floor


def build_static_tables():
    td = sum(RAW_DIGRAMS.values())
    tt = sum(RAW_TRIGRAMS.values())
    dg = {k: math.log(v / td) for k, v in RAW_DIGRAMS.items()}
    tg = {k: math.log(v / tt) for k, v in RAW_TRIGRAMS.items()}
    return dg, math.log(0.01 / td), tg, math.log(0.01 / tt)


def build_corpus_tables(text: str):
    bg, tg = {}, {}
    txt = "".join(c for c in text.lower() if c in ALPHA_SET)
    for i in range(len(txt) - 1):
        bg[txt[i:i + 2]] = bg.get(txt[i:i + 2], 0) + 1
    for i in range(len(txt) - 2):
        tg[txt[i:i + 3]] = tg.get(txt[i:i + 3], 0) + 1
    td, tt = sum(bg.values()), sum(tg.values())
    bg_lp = {k: math.log(v / td) for k, v in bg.items()}
    tg_lp = {k: math.log(v / tt) for k, v in tg.items()}
    return bg_lp, math.log(0.01 / max(td, 1)), tg_lp, math.log(0.01 / max(tt, 1))


# --- Hill Climbing ---

def _score_bigram(text, table, floor):
    return sum(table.get(text[i:i + 2], floor) for i in range(len(text) - 1))


def hill_climbing_solve(cipher_text, bigram_table, floor,
                        max_iterations=8000, restarts=3, seed=0):
    rng = random.Random(seed)
    chars = list(ALPHABET)
    best_plain, best_score = cipher_text, float("-inf")

    for _ in range(restarts):
        shuffled = chars.copy()
        rng.shuffle(shuffled)
        key = {c: s for c, s in zip(chars, shuffled)}
        cur_plain = apply_key(cipher_text, key)
        cur_score = _score_bigram(cur_plain, bigram_table, floor)

        for _ in range(max_iterations):
            i, j = rng.sample(range(len(chars)), 2)
            c1, c2 = chars[i], chars[j]
            key[c1], key[c2] = key[c2], key[c1]
            new_plain = apply_key(cipher_text, key)
            new_score = _score_bigram(new_plain, bigram_table, floor)
            if new_score > cur_score:
                cur_score, cur_plain = new_score, new_plain
            else:
                key[c1], key[c2] = key[c2], key[c1]

        if cur_score > best_score:
            best_score, best_plain = cur_score, cur_plain
    return best_plain, best_score


# --- Simulated Annealing ---

def _score_ngram(text, tg_lp, floor_t, bg_lp, floor_b, w_tri=0.7, w_bi=0.3):
    s = 0.0
    for i in range(len(text) - 2):
        s += w_tri * tg_lp.get(text[i:i + 3], floor_t)
    for i in range(len(text) - 1):
        s += w_bi * bg_lp.get(text[i:i + 2], floor_b)
    return s


def simulated_annealing_solve(cipher_text, tg_lp, floor_t, bg_lp, floor_b,
                              max_iter=15000, T_start=10.0, T_end=0.1,
                              restarts=2, rng_seed=0):
    rng = random.Random(rng_seed)
    chars = list(ALPHABET)
    cipher_freq = {}
    for c in cipher_text:
        if c in ALPHA_SET:
            cipher_freq[c] = cipher_freq.get(c, 0) + 1

    best_plain, best_score = cipher_text, float("-inf")

    for restart in range(restarts):
        if restart == 0:  # frequency-based smart start
            key, used = {}, set()
            for cc in sorted(cipher_freq, key=cipher_freq.get, reverse=True):
                for tc in TR_ORDER:
                    if tc not in used:
                        key[cc] = tc; used.add(tc); break
            for c in chars:
                if c not in key:
                    for tc in TR_ORDER:
                        if tc not in used:
                            key[c] = tc; used.add(tc); break
        else:
            shuffled = chars.copy()
            rng.shuffle(shuffled)
            key = {c: s for c, s in zip(chars, shuffled)}

        plain = apply_key(cipher_text, key)
        cur_score = _score_ngram(plain, tg_lp, floor_t, bg_lp, floor_b)
        decay = (T_end / T_start) ** (1.0 / max(max_iter - 1, 1))
        T = T_start

        for _ in range(max_iter):
            i, j = rng.sample(range(len(chars)), 2)
            c1, c2 = chars[i], chars[j]
            key[c1], key[c2] = key[c2], key[c1]
            new_plain = apply_key(cipher_text, key)
            new_score = _score_ngram(new_plain, tg_lp, floor_t, bg_lp, floor_b)
            delta = new_score - cur_score
            if delta > 0 or rng.random() < math.exp(delta / T):
                cur_score, plain = new_score, new_plain
            else:
                key[c1], key[c2] = key[c2], key[c1]
            T *= decay

        if cur_score > best_score:
            best_score, best_plain = cur_score, plain
    return best_plain, best_score
