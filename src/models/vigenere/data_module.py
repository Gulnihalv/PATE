"""Vigenère data module."""

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from generators.vigenere_generator import VigenereKeystreamGenerator


class VigenereDataModule(pl.LightningDataModule):
    def __init__(self, text_path, alphabet, batch_size=64,
                 min_key_len=3, max_key_len=12, val_split=0.1):
        super().__init__()
        self.text_path = text_path
        self.alphabet = alphabet
        self.batch_size = batch_size
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len
        self.val_split = val_split

    def setup(self, stage=None):
        # training dataset (curriculum learning applied)
        self.train_full = VigenereKeystreamGenerator(
            text_path=self.text_path,
            alphabet=self.alphabet,
            min_key_len=self.min_key_len,
            max_key_len=self.max_key_len,
        )

        self.val_full = VigenereKeystreamGenerator(
            text_path=self.text_path,
            alphabet=self.alphabet,
            min_key_len=self.min_key_len,
            max_key_len=self.max_key_len,
        )
        # fixed chunk-size distribution for validation
        self.val_full.CHUNK_WEIGHTS = [0.33, 0.34, 0.33]

        total_len = len(self.train_full)
        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len

        self.train_dataset, _ = torch.utils.data.random_split(
            self.train_full, [train_len, val_len]
        )
        self.val_dataset, _ = torch.utils.data.random_split(
            self.val_full, [val_len, train_len]
        )

        self.dataset = self.train_full

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
        )
