import random
import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100  # target value at padded positions (ignored by loss/metrics)


class SubstutionDataGenerator(Dataset):
    """Space-free substitution data. Vocabulary = the alphabet only (indices 0..N-1).

    Padding is not a token: each item carries a boolean pad mask instead, and
    padded target positions are IGNORE_INDEX.
    """

    def __init__(self, text_path, alphabet, seq_len):
        super().__init__()

        self.alphabet = list(alphabet)
        self.vocab_size = len(self.alphabet)
        self.seq_len = seq_len
        self.char2idx = {c: i for i, c in enumerate(self.alphabet)}
        self.idx2char = {i: c for i, c in enumerate(self.alphabet)}

        with open(text_path, "r", encoding="utf-8") as f:
            raw = f.read()
        # Turkish-aware lowercase, then drop everything outside the alphabet (spaces, punctuation, digits)
        raw = raw.replace("I", "\u0131").replace("\u0130", "i").lower()
        self.text = "".join(c for c in raw if c in self.char2idx)

        self.text_len = len(self.text)
        self.chunks = [(p, min(p + seq_len, self.text_len))
                       for p in range(0, self.text_len, seq_len)]
        print(f"total chunks: {len(self.chunks)}")

    def __len__(self):
        return len(self.chunks)

    def generate_random_key(self):
        shuffled = list(self.alphabet)
        random.shuffle(shuffled)
        return dict(zip(self.alphabet, shuffled))

    def str_to_indices(self, text):
        return [self.char2idx[c] for c in text]

    def get_chunk(self, index):
        start, end = self.chunks[index]
        return self.text[start:end]

    def __getitem__(self, index):
        plain_text = self.get_chunk(index)

        # Dynamic cropping: ~50% of the time use a random-length window
        if len(plain_text) > 40 and random.random() < 0.5:
            n = random.randint(30, len(plain_text) - 1)
            s = random.randint(0, len(plain_text) - n)
            plain_text = plain_text[s:s + n]

        keymap = self.generate_random_key()
        cipher_text = plain_text.translate(str.maketrans(keymap))

        n = len(plain_text)
        pad = self.seq_len - n
        src = self.str_to_indices(cipher_text) + [0] * pad
        tgt = self.str_to_indices(plain_text) + [IGNORE_INDEX] * pad
        pad_mask = [False] * n + [True] * pad

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(pad_mask, dtype=torch.bool),
            torch.tensor(tgt, dtype=torch.long),
        )
