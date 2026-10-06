"""Classical (learning-free) Vigenere solver used as a baseline.

Period estimation proceeds in three steps:

  1. a Kasiski divisor vote over the distances between repeated n-grams,
  2. a coset coincidence search over the whole candidate range,
  3. a reduction of the winner to its fundamental period.

Step 3 is required because the coset coincidence rate is invariant under
multiples of the true period: if the period is d, then partitioning by any
multiple of d also yields pure Caesar cosets and therefore the same expected
coincidence rate. The minimiser of the coincidence criterion is consequently
not necessarily the fundamental period, and without step 3 the solver reports
a multiple of d whenever one lies inside the candidate range.
"""

import math
from collections import Counter

ALPHABET_CHARS = "abcçdefgğhıijklmnoöprsştuüvyz"   # 29 Turkish letters (no space)
ALPHA = list(ALPHABET_CHARS)
ALPHA_S = set(ALPHA)
N_ALPHA = len(ALPHA)                                # 29
TR_IOC = 0.0762
UNIFORM_IOC = 1.0 / N_ALPHA                         # 0.0345

MIN_KEY_LEN = 3
MAX_KEY_LEN = 32          # must match the experimental key-length range

# a coset needs at least this many characters for its coincidence rate to be
# estimated at all; candidates that would leave shorter cosets are not searched
MIN_COSET_LEN = 2

# a divisor is accepted as the fundamental period when its coset coincidence
# rate is at least this fraction of the winning candidate's
FUNDAMENTAL_TOL = 0.90

# Turkish letter frequencies (normalized, space-free alphabet)
TR_FREQ_VIG = {
    'a': 0.1281, 'e': 0.0994, 'i': 0.0916, 'n': 0.0815, 'l': 0.0606,
    'r': 0.0744, 't': 0.0578, 'k': 0.0581, 'm': 0.0469, 's': 0.0377,
    'y': 0.0399, 'd': 0.0481, 'u': 0.0363, 'o': 0.0282, 'z': 0.0164,
    'b': 0.0316, 'ş': 0.0209, 'ç': 0.0123, 'g': 0.0131, 'ğ': 0.0123,
    'h': 0.0125, 'ü': 0.0199, 'ö': 0.0102, 'p': 0.0086, 'ı': 0.0573,
    'c': 0.0092, 'f': 0.0039, 'v': 0.0102, 'j': 0.0002,
}

CHAR2IDX = {c: i for i, c in enumerate(ALPHA)}


def index_of_coincidence(text: str) -> float:
    n = len(text)
    if n < 2:
        return 0.0
    counts = Counter(text)
    return sum(c * (c - 1) for c in counts.values()) / (n * (n - 1))


def _avg_group_ioc(cipher_text: str, k: int) -> float:
    """Mean of the per-coset indices of coincidence induced by period k."""
    groups = [cipher_text[i::k] for i in range(k)]
    valid = [g for g in groups if len(g) >= MIN_COSET_LEN]
    if not valid:
        return 0.0
    return sum(index_of_coincidence(g) for g in valid) / len(valid)


def _pooled_group_ioc(cipher_text: str, k: int) -> float:
    """Coincidence rate with counts pooled across the k cosets.

    Pooling the matching-pair counts rather than averaging the per-coset
    indices gives the same expectation with a lower variance, which matters
    when the cosets are short.
    """
    num = den = 0
    for i in range(k):
        g = cipher_text[i::k]
        m = len(g)
        if m < MIN_COSET_LEN:
            continue
        counts = Counter(g)
        num += sum(c * (c - 1) for c in counts.values())
        den += m * (m - 1)
    return num / den if den else 0.0


def searchable_range(cipher_text, min_k=MIN_KEY_LEN, max_k=MAX_KEY_LEN):
    """The candidate periods actually reachable for this text length.

    A candidate k is only searched when it leaves at least MIN_COSET_LEN
    characters per coset, so on short texts the effective upper bound is
    below max_k. Reported alongside the results so that the search range is
    never implicit.
    """
    hi = min(max_k, len(cipher_text) // MIN_COSET_LEN)
    return min_k, hi


def _reduce_to_fundamental(cipher_text, k, min_k=MIN_KEY_LEN,
                           tol=FUNDAMENTAL_TOL, score=_pooled_group_ioc):
    """Replace k by the smallest divisor whose cosets are as coherent as k's.

    Multiples of the true period score as well as the period itself, so the
    winner of the coincidence search is reduced here. A divisor of the true
    period mixes several shifts within each coset and scores near the uniform
    rate, so it fails the test.
    """
    ref = score(cipher_text, k)
    if ref <= 0.0:
        return k
    for d in range(min_k, k):
        if k % d != 0:
            continue
        if len(cipher_text) < MIN_COSET_LEN * d:
            continue
        if score(cipher_text, d) >= tol * ref:
            return d
    return k


def estimate_key_length_ioc(cipher_text, min_k=MIN_KEY_LEN, max_k=MAX_KEY_LEN,
                            tol=FUNDAMENTAL_TOL, score=_pooled_group_ioc) -> int:
    """Coset coincidence search, then reduction to the fundamental period.

    Pass tol=None to return the raw minimiser without the reduction step; this
    is the ablation reported as the unreduced estimator.
    """
    lo, hi = searchable_range(cipher_text, min_k, max_k)
    best_k, best_dist = lo, float("inf")
    for k in range(lo, hi + 1):
        dist = abs(score(cipher_text, k) - TR_IOC)
        if dist < best_dist:
            best_dist, best_k = dist, k
    if tol is None:
        return best_k
    return _reduce_to_fundamental(cipher_text, best_k, min_k, tol, score)


def _ngram_distances(cipher_text, ngram_len=3, max_pairs=20000):
    """Distances between every pair of occurrences of each repeated n-gram."""
    positions = {}
    for i in range(len(cipher_text) - ngram_len + 1):
        positions.setdefault(cipher_text[i:i + ngram_len], []).append(i)

    distances = []
    for pos_list in positions.values():
        if len(pos_list) < 2:
            continue
        for a in range(len(pos_list)):
            for b in range(a + 1, len(pos_list)):
                distances.append(pos_list[b] - pos_list[a])
                if len(distances) >= max_pairs:
                    return distances
    return distances


def estimate_key_length_kasiski(cipher_text, min_k=MIN_KEY_LEN, max_k=MAX_KEY_LEN,
                                ngram_len=3, min_distances=8, min_lift=1.5):
    """Kasiski examination by divisor vote.

    Each candidate k is scored by how much more often it divides the observed
    n-gram distances than chance would predict: a fraction 1/k of arbitrary
    distances is divisible by k, so the lift is (observed fraction) * k. The
    global gcd of the distances is not used, because a handful of coincidental
    repeats at coprime distances drives it to one and discards the evidence.

    Returns None when the text yields too few repeats, or when no candidate
    stands out, in which case the caller should rely on the coincidence search.
    """
    distances = [d for d in _ngram_distances(cipher_text, ngram_len) if d > 0]
    if len(distances) < min_distances:
        return None

    lo, hi = searchable_range(cipher_text, min_k, max_k)
    total = len(distances)
    best_k, best_lift = None, 0.0
    for k in range(lo, hi + 1):
        hits = sum(1 for d in distances if d % k == 0)
        lift = (hits / total) * k
        if lift > best_lift:
            best_lift, best_k = lift, k
    if best_k is None or best_lift < min_lift:
        return None
    return best_k


def estimate_key_length_combined(cipher_text, min_k=MIN_KEY_LEN, max_k=MAX_KEY_LEN,
                                 tol=FUNDAMENTAL_TOL, score=_pooled_group_ioc) -> int:
    """Kasiski vote and coincidence search together, reduced to the fundamental."""
    candidates = {estimate_key_length_ioc(cipher_text, min_k, max_k,
                                          tol=None, score=score)}
    kas = estimate_key_length_kasiski(cipher_text, min_k, max_k)
    if kas is not None:
        candidates.add(kas)

    best_k, best_dist = min(candidates), float("inf")
    for k in sorted(candidates):
        dist = abs(score(cipher_text, k) - TR_IOC)
        if dist < best_dist:
            best_dist, best_k = dist, k

    if tol is None:
        return best_k
    return _reduce_to_fundamental(cipher_text, best_k, min_k, tol, score)


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
            c_idx = CHAR2IDX[char]
            p_idx = (c_idx - key_shifts[pos % k]) % N_ALPHA
            result.append(ALPHA[p_idx])
            pos += 1
        else:
            result.append(char)
    return "".join(result)


def classical_vigenere_solve(cipher_text, min_k=MIN_KEY_LEN, max_k=MAX_KEY_LEN,
                             true_key_len=None, tol=FUNDAMENTAL_TOL):
    """Full classical pipeline.

    With true_key_len supplied the period estimation stage is skipped, which
    isolates shift recovery from period recovery.
    Returns (plaintext, used_key_len, key_shifts).
    """
    if true_key_len is not None:
        pred_key_len = true_key_len
    else:
        pred_key_len = estimate_key_length_combined(cipher_text, min_k, max_k, tol=tol)

    groups = [cipher_text[i::pred_key_len] for i in range(pred_key_len)]
    key_shifts = [crack_caesar_position(g) for g in groups]
    plaintext = vigenere_decrypt(cipher_text, key_shifts)
    return plaintext, pred_key_len, key_shifts
