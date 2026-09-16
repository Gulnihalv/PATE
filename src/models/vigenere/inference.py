"""Vigenère inference helpers: vocabulary, encryption, tensor conversion.

Vocabulary: alphabet indices 0..28, PAD = vocab_size (29). The model input
contains no spaces/special characters; text is cleaned against the alphabet first.
"""

import torch


def build_vocab(alphabet_chars: str):
    char2idx = {c: i for i, c in enumerate(alphabet_chars)}
    idx2char = {i: c for i, c in enumerate(alphabet_chars)}
    return char2idx, idx2char


def encrypt_vigenere(plain_text, key, char2idx, idx2char):
    """Encrypt with Vigenère, dropping characters outside the alphabet.
    Returns (cipher_text, cleaned_plain_text)."""
    vocab_size = len(char2idx)
    clean = "".join(c for c in plain_text.lower() if c in char2idx)
    key_indices = [char2idx[k] for k in key]
    cipher = [
        idx2char[(char2idx[c] + key_indices[i % len(key_indices)]) % vocab_size]
        for i, c in enumerate(clean)
    ]
    return "".join(cipher), clean


def text_to_tensor(cipher_text, char2idx, device="cpu"):
    return torch.tensor([char2idx[c] for c in cipher_text], dtype=torch.long, device=device)
