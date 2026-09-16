"""Substitution data module, shared by the BTE and FATE models.

Wraps SubstutionDataGenerator and produces a fixed (seeded) train/validation split.
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from generators.substitution_generator import SubstutionDataGenerator


class SubstitutionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        text_path: str,
        alphabet: str,
        seq_len: int = 256,
        batch_size: int = 128,
        val_split: float = 0.1,
        num_workers: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vocab_size = None  # set in setup()

    def setup(self, stage=None):
        full = SubstutionDataGenerator(
            self.hparams.text_path, self.hparams.alphabet, self.hparams.seq_len
        )
        self.vocab_size = len(full.full_alphabet)  # 29 letters + 4 special tokens = 33

        total = len(full)
        val_size = int(total * self.hparams.val_split)
        train_size = total - val_size

        g = torch.Generator().manual_seed(self.hparams.seed)
        self.train_dataset, self.val_dataset = random_split(
            full, [train_size, val_size], generator=g
        )
        print(f"Total: {total:,} | Train: {train_size:,} | Val: {val_size:,}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,
            num_workers=self.hparams.num_workers, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False,
            num_workers=self.hparams.num_workers, pin_memory=True,
        )
