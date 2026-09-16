"""Classical (learning-free) Vigenère solver used as a baseline.

Pipeline:
  1. Key-length estimation: Kasiski (GCD of repeated-trigram distances) plus
     Index of Coincidence (target Turkish IoC ~0.0762); the candidate closest
     to the target IoC is chosen.
  2. Per key position, the Caesar shift is found by chi-square against Turkish
     letter frequencies.
  3. Vigenère decryption.

Model-independent (standard library only).
"""

import math
from collections import Counter
from functools import reduce

ALPHABET_CHARS = "abcçdefgğhıijklmnoöprsştuüvyz"   # 29 Turkish letters (no space)
ALPHA = list(ALPHABET_CHARS)
ALPHA_S = set(ALPHA)
N_ALPHA = len(ALPHA)                                # 29
TR_IOC = 0.0762

# Turkish letter frequencies (normalized, space-free alphabet)
TR_FREQ_VIG = {
    'a': 0.1281, 'e': 0.0994, 'i': 0.0916, 'n': 0.0815, 'l': 0.0606,
    'r': 0.0744, 't': 0.0578, 'k': 0.0581, 'm': 0.0469, 's': 0.0377,
    'y': 0.0399, 'd': 0.0481, 'u': 0.0363, 'o': 0.0282, 'z': 0.0164,
    'b': 0.0316, 'ş': 0.0209, 'ç': 0.0123, 'g': 0.0131, 'ğ': 0.0123,
    'h': 0.0125, 'ü': 0.0199, 'ö': 0.0102, 'p': 0.0086, 'ı': 0.0573,
    'c': 0.0092, 'f': 0.0039, 'v': 0.0102, 'j': 0.0002,
}


def index_of_coincidence(text: str) -> float:
    n = len(text)
    if n < 2:
        return 0.0
    counts = Counter(text)
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def estimate_key_length_ioc(cipher_text, min_k=3, max_k=12) -> int:
    best_k, best_dist = min_k, float("inf")
    for k in range(min_k, max_k + 1):
        groups = ["".join(cipher_text[i::k]) for i in range(k)]
        avg_ioc = sum(index_of_coincidence(g) for g in groups) / k
        dist = abs(avg_ioc - TR_IOC)
        if dist < best_dist:
            best_dist, best_k = dist, k
    return best_k


def estimate_key_length_kasiski(cipher_text, min_k=3, max_k=12, ngram_len=3) -> int:
    positions = {}
    for i in range(len(cipher_text) - ngram_len + 1):
        ng = cipher_text[i:i + ngram_len]
        positions.setdefault(ng, []).append(i)

    distances = []
    for pos_list in positions.values():
        if len(pos_list) > 1:
            for j in range(1, len(pos_list)):
                distances.append(pos_list[j] - pos_list[j - 1])

    if not distances:
        return estimate_key_length_ioc(cipher_text, min_k, max_k)

    common_gcd = reduce(math.gcd, distances)
    if min_k <= common_gcd <= max_k:
        return common_gcd
    return estimate_key_length_ioc(cipher_text, min_k, max_k)


def estimate_key_length_combined(cipher_text, min_k=3, max_k=12) -> int:
    candidates = {estimate_key_length_ioc(cipher_text, min_k, max_k),
                  estimate_key_length_kasiski(cipher_text, min_k, max_k)}
    best_k, best_sc = min(candidates), float("inf")
    for k in candidates:
        groups = ["".join(cipher_text[i::k]) for i in range(k)]
        avg_ioc = sum(index_of_coincidence(g) for g in groups) / k
        sc = abs(avg_ioc - TR_IOC)
        if sc < best_sc:
            best_sc, best_k = sc, k
    return best_k


def crack_caesar_position(group: str) -> int:
    """Solve a single Caesar group by chi-square against Turkish frequencies."""
    if not group:
        return 0
    n = len(group)
    counts = Counter(group)
    best_shift, best_chi_sq = 0, float("inf")
    for shift in range(N_ALPHA):
        chi_sq = 0.0
        for idx, char in enumerate(ALPHA):
            shifted_char = ALPHA[(idx + shift) % N_ALPHA]
            observed = counts.get(shifted_char, 0) / n
            expected = TR_FREQ_VIG.get(char, 1 / N_ALPHA)
            chi_sq += (observed - expected) ** 2 / expected
        if chi_sq < best_chi_sq:
            best_chi_sq, best_shift = chi_sq, shift
    return best_shift


def vigenere_decrypt(cipher_text, key_shifts) -> str:
    k = len(key_shifts)
    result, pos = [], 0
    for char in cipher_text:
        if char in ALPHA_S:
            c_idx = ALPHA.index(char)
            p_idx = (c_idx - key_shifts[pos % k]) % N_ALPHA
            result.append(ALPHA[p_idx])
            pos += 1
        else:
            result.append(char)
    return "".join(result)


def classical_vigenere_solve(cipher_text, min_k=3, max_k=12):
    """Full classical pipeline. Returns (plaintext, estimated_key_length)."""
    pred_key_len = estimate_key_length_combined(cipher_text, min_k, max_k)
    groups = ["".join(cipher_text[i::pred_key_len]) for i in range(pred_key_len)]
    key_shifts = [crack_caesar_position(g) for g in groups]
    plaintext = vigenere_decrypt(cipher_text, key_shifts)
    return plaintext, pred_key_len
