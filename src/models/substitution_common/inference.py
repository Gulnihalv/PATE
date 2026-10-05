"""Substitution inference: vocabulary and sliding-window decoding (space-free).

For texts longer than seq_len a sliding window is used. Two modes:
  - raw        : each position takes the argmax of the first window covering it
  - consistent : a single bijective cipher->plain map is built from global votes

Vocabulary matches the generator: the alphabet only (letter i -> index i, no
special tokens). Padding is passed to the model as a boolean mask.
"""

from collections import defaultdict
import torch


def clean_text(text: str, alphabet: str) -> str:
    """Turkish-aware lowercase and drop every character outside the alphabet."""
    text = text.replace("I", "\u0131").replace("\u0130", "i").lower()
    keep = set(alphabet)
    return "".join(c for c in text if c in keep)


def build_vocab(alphabet: str):
    char2idx = {c: i for i, c in enumerate(alphabet)}
    idx2char = {i: c for i, c in enumerate(alphabet)}
    return char2idx, idx2char


def text_to_padded_tensor(text: str, char2idx: dict, seq_len: int):
    """Return (src [seq_len], pad_mask [seq_len], True = padding)."""
    idx = [char2idx[c] for c in text][:seq_len]
    n = len(idx)
    idx += [0] * (seq_len - n)
    mask = [False] * n + [True] * (seq_len - n)
    return torch.tensor(idx, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def _batch(text, char2idx, seq_len, device):
    src, mask = text_to_padded_tensor(text, char2idx, seq_len)
    return src.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)


def decode_tensor(pred_tensor, pad_mask, idx2char) -> str:
    preds = pred_tensor.squeeze(0).tolist()
    pads = pad_mask.squeeze(0).tolist()
    return "".join(idx2char[p] for p, is_pad in zip(preds, pads) if not is_pad)


def build_consistent_map(votes: dict) -> dict:
    """Build a bijective cipher->plain map from global vote counts."""
    final_map, used = {}, set()
    sorted_ciphers = sorted(votes.keys(), key=lambda k: sum(votes[k].values()), reverse=True)
    for c_idx in sorted_ciphers:
        for cand, _ in sorted(votes[c_idx].items(), key=lambda x: x[1], reverse=True):
            if cand not in used:
                final_map[c_idx] = cand
                used.add(cand)
                break
    return final_map


def apply_consistent_map(cipher_text, final_map, char2idx, idx2char) -> str:
    out = []
    for ch in cipher_text:
        c_idx = char2idx[ch]
        out.append(idx2char[final_map[c_idx]] if c_idx in final_map else ch)
    return "".join(out)


def _nn_module(model):
    """Return the inner nn.Module of a Lightning wrapper, or the model itself."""
    return model.model if hasattr(model, "model") else model


@torch.no_grad()
def infer_raw(model, cipher_text, char2idx, idx2char,
              seq_len=256, stride=200, device="cpu") -> str:
    net = _nn_module(model)
    n = len(cipher_text)

    if n <= seq_len:
        src, mask = _batch(cipher_text, char2idx, seq_len, device)
        return decode_tensor(net.generate(src, mask), mask, idx2char)

    raw = [""] * n
    start = 0
    while start < n:
        end = min(start + seq_len, n)
        window = cipher_text[start:end]
        src, mask = _batch(window, char2idx, seq_len, device)
        pred = net.generate(src, mask).squeeze(0).tolist()
        for i in range(len(window)):
            g = start + i
            if raw[g] == "":
                raw[g] = idx2char[pred[i]]
        if end == n:
            break
        start += stride
    return "".join(raw)


@torch.no_grad()
def infer_consistent(model, cipher_text, char2idx, idx2char,
                     seq_len=256, stride=200, device="cpu") -> str:
    net = _nn_module(model)
    n = len(cipher_text)

    votes = defaultdict(lambda: defaultdict(int))
    start = 0
    while start < n:
        end = min(start + seq_len, n)
        window = cipher_text[start:end]
        src, mask = _batch(window, char2idx, seq_len, device)
        src_list = src.squeeze(0).tolist()
        pred_list = net.generate(src, mask).squeeze(0).tolist()
        for i in range(len(window)):
            votes[src_list[i]][pred_list[i]] += 1
        if end == n:
            break
        start += stride
    return apply_consistent_map(cipher_text, build_consistent_map(votes), char2idx, idx2char)
