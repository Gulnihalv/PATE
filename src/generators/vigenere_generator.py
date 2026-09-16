import random
import torch
from torch.utils.data import Dataset

class VigenereKeystreamGenerator(Dataset):
    CHUNK_SIZES   = [128, 256, 512]
    CHUNK_WEIGHTS = [0.2, 0.6, 0.2]

    def __init__(self, text_path, alphabet, min_key_len=3, max_key_len=12):
        super().__init__()

        self.crypto_alphabet   = list(alphabet)
        self.crypto_vocab_size = len(self.crypto_alphabet)
        self.total_vocab_size  = self.crypto_vocab_size + 1

        self.int2char = {i: c for i, c in enumerate(self.crypto_alphabet)}
        self.char2int = {c: i for i, c in enumerate(self.crypto_alphabet)}

        self.PAD_TOKEN_IDX = self.crypto_vocab_size
        self.seq_len     = max(self.CHUNK_SIZES)
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len

        with open(text_path, "r", encoding="utf-8") as f:
            raw_text = f.read().replace("\n", " ").strip()
            allowed   = set(self.crypto_alphabet)
            self.text = "".join([c for c in raw_text if c in allowed])

        self.text_len = len(self.text)
        self.chunk_offsets = self._compute_all_offsets()
        self._len = len(self.chunk_offsets[512])

    def _compute_all_offsets(self):
        offsets = {}
        for size in self.CHUNK_SIZES:
            starts = list(range(0, self.text_len - size + 1, size))
            offsets[size] = starts
        return offsets

    def __len__(self):
        return self._len

    def _pick_chunk_size(self):
        return random.choices(self.CHUNK_SIZES, weights=self.CHUNK_WEIGHTS, k=1)[0]

    def generate_random_key(self):
        k_len      = random.randint(self.min_key_len, self.max_key_len)
        key_indices = [random.randint(0, self.crypto_vocab_size - 1) for _ in range(k_len)]
        return key_indices, k_len

    def encrypt_vigenere(self, text_chunk, key_indices):
        src_indices    = []
        tgt_indices    = []
        keystream_target = []
        cyclic_positions = []

        key_len  = len(key_indices)
        key_ptr  = 0

        for char in text_chunk:
            if char not in self.char2int:
                continue

            p_val       = self.char2int[char]
            current_key = key_indices[key_ptr % key_len]
            c_val       = (p_val + current_key) % self.crypto_vocab_size

            src_indices.append(c_val)
            tgt_indices.append(p_val)
            keystream_target.append(current_key)
            cyclic_positions.append(key_ptr % key_len)

            key_ptr += 1

        return src_indices, tgt_indices, keystream_target, cyclic_positions

    def __getitem__(self, index):
        chunk_size = self._pick_chunk_size()

        offsets    = self.chunk_offsets[chunk_size]
        start      = offsets[index % len(offsets)]
        text_chunk = self.text[start: start + chunk_size]

        # keep at least max_key_len * 5 characters
        min_safe = self.max_key_len * 5
        if len(text_chunk) > min_safe and random.random() < 0.20:
            new_len    = random.randint(min_safe, len(text_chunk))
            text_chunk = text_chunk[:new_len]

        key_indices, k_len = self.generate_random_key()
        src, tgt, key_target, cyc_pos = self.encrypt_vigenere(text_chunk, key_indices)

        pad_len = self.seq_len - len(src)

        src_padded        = src        + [self.PAD_TOKEN_IDX] * pad_len
        key_target_padded = key_target + [self.PAD_TOKEN_IDX] * pad_len
        tgt_padded        = tgt        + [self.PAD_TOKEN_IDX] * pad_len
        cyc_pos_padded    = cyc_pos    + [k_len]              * pad_len

        return {
            "src":        torch.tensor(src_padded,        dtype=torch.long),
            "key_target": torch.tensor(key_target_padded, dtype=torch.long),
            "tgt_plain":  torch.tensor(tgt_padded,        dtype=torch.long),
            "cycle_pos":  torch.tensor(cyc_pos_padded,    dtype=torch.long),
            "key_len":    torch.tensor(k_len,             dtype=torch.long),
            "chunk_size": torch.tensor(chunk_size,        dtype=torch.long), 
        }