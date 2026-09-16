"""Substitution inference: vocabulary and sliding-window decoding.

For texts longer than seq_len a sliding window is used. Two modes:
  - raw        : each position takes the argmax of the first window covering it
  - consistent : a single bijective cipher->plain map is built from global votes

Vocabulary order matches the generator: [<PAD>, <SOS>, <EOS>, space] + alphabet
(PAD=0, SOS=1, EOS=2, space=3, letters from index 4).
"""

from collections import defaultdict
import torch

PAD_IDX, SOS_IDX, EOS_IDX, SPACE_IDX = 0, 1, 2, 3
SPECIAL_TOKENS = ["<PAD>", "<SOS>", "<EOS>", " "]


def build_vocab(alphabet: str):
    full = SPECIAL_TOKENS + list(alphabet)
    char2idx = {c: i for i, c in enumerate(full)}
    idx2char = {i: c for i, c in enumerate(full)}
    return char2idx, idx2char


def text_to_padded_tensor(text: str, char2idx: dict, seq_len: int) -> torch.Tensor:
    idx = [char2idx.get(c, PAD_IDX) for c in text][:seq_len]
    idx += [PAD_IDX] * (seq_len - len(idx))
    return torch.tensor(idx, dtype=torch.long)


def decode_tensor(pred_tensor, src_tensor, idx2char) -> str:
    preds = pred_tensor.squeeze().tolist()
    srcs = src_tensor.squeeze().tolist()
    out = []
    for p_idx, s_idx in zip(preds, srcs):
        if s_idx == PAD_IDX:
            break
        ch = idx2char.get(p_idx, "")
        if ch in ("<PAD>", "<EOS>"):
            break
        if ch == "<SOS>":
            continue
        out.append(ch)
    return "".join(out)


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
        c_idx = char2idx.get(ch, PAD_IDX)
        out.append(idx2char.get(final_map[c_idx], ch) if c_idx in final_map else ch)
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
        src = text_to_padded_tensor(cipher_text, char2idx, seq_len).unsqueeze(0).to(device)
        return decode_tensor(net.generate(src), src, idx2char)

    raw = [""] * n
    start = 0
    while start < n:
        end = min(start + seq_len, n)
        window = cipher_text[start:end]
        src = text_to_padded_tensor(window, char2idx, seq_len).unsqueeze(0).to(device)
        pred = net.generate(src).squeeze().tolist()
        for i in range(len(window)):
            g = start + i
            if raw[g] == "":
                ch = idx2char.get(pred[i], "")
                raw[g] = window[i] if ch in SPECIAL_TOKENS else ch
        if end == n:
            break
        start += stride
    return "".join(raw)


@torch.no_grad()
def infer_consistent(model, cipher_text, char2idx, idx2char,
                     seq_len=256, stride=200, device="cpu") -> str:
    net = _nn_module(model)
    n = len(cipher_text)

    if n <= seq_len:
        src = text_to_padded_tensor(cipher_text, char2idx, seq_len).unsqueeze(0).to(device)
        pred = net.generate(src)
        cipher_seq = src.squeeze().tolist()
        pred_seq = pred.squeeze().tolist()
        votes = {}
        for c, p in zip(cipher_seq, pred_seq):
            if c < 4:
                continue
            votes.setdefault(c, {}).setdefault(p, 0)
            votes[c][p] += 1
        return apply_consistent_map(cipher_text, build_consistent_map(votes), char2idx, idx2char)

    global_votes = defaultdict(lambda: defaultdict(int))
    start = 0
    while start < n:
        end = min(start + seq_len, n)
        window = cipher_text[start:end]
        src = text_to_padded_tensor(window, char2idx, seq_len).unsqueeze(0).to(device)
        src_list = src.squeeze().tolist()
        pred_list = net.generate(src).squeeze().tolist()
        for i in range(len(window)):
            c_idx, p_idx = src_list[i], pred_list[i]
            if c_idx >= 4:
                global_votes[c_idx][p_idx] += 1
        if end == n:
            break
        start += stride
    return apply_consistent_map(cipher_text, build_consistent_map(global_votes), char2idx, idx2char)
