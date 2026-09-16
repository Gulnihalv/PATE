import random
import torch
from torch.utils.data import Dataset

class SubstutionDataGenerator(Dataset):
    def __init__(self, text_path, alphabet, seq_len):
        super().__init__()

        self.PAD_TOKEN = "<PAD>"
        self.SOS_TOKEN = "<SOS>"
        self.EOS_TOKEN = "<EOS>"
        self.space = " "

        self.special_tokens = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.space]
        self.full_alphabet = self.special_tokens + list(alphabet)

        self.seq_len = seq_len
        self.char2idx = {c: i for i, c in enumerate(self.full_alphabet)}
        self.idx2char = {i: c for i, c in enumerate(self.full_alphabet)}

        with open(text_path, "r", encoding="utf-8") as f:
            self.text = f.read().replace("\n", " ")

        self.text_len = len(self.text)
        # reserve room for SOS and EOS
        self.content_len = self.seq_len - 2 

        self.chunks = self._compute_optimized_chunks() 
        print(f"total chunks: {len(self.chunks)}")

    def _compute_optimized_chunks(self):
        chunks = []
        pos = 0

        while pos < self.text_len:
            end_pos = min(pos + self.content_len, self.text_len)

            # last chunk: add directly
            if end_pos >= self.text_len:
                chunks.append((pos, end_pos))
                break

            # find the last space so words are not split
            chunk_str = self.text[pos: end_pos]
            last_space_idx = chunk_str.rfind(' ')

            if last_space_idx != -1:
                # take up to the space (+1 to include the space itself)
                end_pos = pos + last_space_idx + 1
            
            chunks.append((pos, end_pos))
            pos = end_pos

        return chunks
    
    def __len__(self):
        return len(self.chunks)
    
    def generate_random_key(self):
        real_chars = self.full_alphabet[4:]
        shuffled = list(real_chars)
        random.shuffle(shuffled)
        return {src: target for src, target in zip(real_chars, shuffled)}
    
    def str_to_indices(self, text):
        return [self.char2idx.get(c, 0) for c in text]
    
    def get_chunk(self, index):
        start, end = self.chunks[index]
        return self.text[start:end]
    
    def __getitem__(self, index):
        # 1. take the original (word-aligned) chunk
        plain_text_raw = self.get_chunk(index)

        # --- Dynamic cropping (preserving word boundaries) ---
        # Strategy: ~50% chance to shorten the text, but never split words.
        
        # only for sufficiently long texts
        if len(plain_text_raw) > 60 and random.random() < 0.5:
            # split into words
            words = plain_text_raw.split(self.space)
            
            # if there are enough words to crop
            if len(words) > 3:
                # pick a random number of words
                # aim for roughly 40-150 characters, but word-based
                min_words = 3
                max_words = len(words)
                
                # random window size (in words)
                num_words_to_take = random.randint(min_words, max_words - 1)
                
                # random start word
                start_word_idx = random.randint(0, len(words) - num_words_to_take)
                
                # join the selected words
                selected_words = words[start_word_idx : start_word_idx + num_words_to_take]
                plain_text = self.space.join(selected_words)
                
                # if cropping left it too short, fall back to the original (safety)
                if len(plain_text) < 10:
                    plain_text = plain_text_raw
            else:
                plain_text = plain_text_raw
        else:
            # no cropping, use the original
            plain_text = plain_text_raw
        # ---

        # Encryption
        keymap = self.generate_random_key()
        table = str.maketrans(keymap)
        cipher_text = plain_text.translate(table)

        # Vectorization
        plain_text_indices = self.str_to_indices(plain_text)
        cipher_text_indices = self.str_to_indices(cipher_text)

        # Input (source)
        src = cipher_text_indices + [self.char2idx[self.PAD_TOKEN]] * (self.seq_len - len(cipher_text_indices))

        # Target: no SOS/EOS, aligned one-to-one
        tgt = plain_text_indices + [self.char2idx[self.PAD_TOKEN]] * (self.seq_len - len(plain_text_indices))

        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )