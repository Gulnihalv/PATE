"""Vanilla transformer encoder (baseline).

No period structure of any kind: token embeddings plus a learned absolute
positional encoding, a standard encoder stack, and a per-position projection to
plaintext. No period conditioning, no key-length head, no validity head. A
single forward pass maps ciphertext to plaintext directly.

This is the unaided arm of the comparison: whatever periodic structure it uses,
it must discover on its own, and it produces no key-length estimate.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class VigenereSolver(pl.LightningModule):
    """Vanilla encoder: ciphertext -> plaintext, one pass, no period machinery."""

    def __init__(self, vocab_size=29, pad_idx=29, max_len=512,
                 min_key_len=3, max_key_len=12,
                 d_model=256, nhead=8, num_layers=6, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.pad_idx     = pad_idx
        self.min_key_len = min_key_len      # kept for eval-code compatibility
        self.max_key_len = max_key_len

        self.char_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)
        self.pos_emb  = nn.Embedding(max_len, d_model)      # learned absolute

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True,
            dropout=0.1, dim_feedforward=2048,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, key_len=None):
        """key_len is accepted and ignored, so shared eval code can call this."""
        B, L = src.shape
        positions = torch.arange(L, device=src.device).unsqueeze(0)
        pad_mask  = (src == self.pad_idx)
        x = self.char_emb(src) + self.pos_emb(positions)
        encoded = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.fc_out(encoded)

    def _shared_step(self, batch):
        src, tgt = batch["src"], batch["tgt_plain"]
        logits = self(src)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=self.pad_idx,
        )
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask  = (tgt != self.pad_idx)
            acc = ((preds == tgt) & mask).sum().float() / mask.sum().clamp(min=1)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc",  acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def decode(self, ciphertext_indices, **kwargs):
        """No period is estimated; best_key_len is reported as None."""
        self.eval()
        logits = self(ciphertext_indices.unsqueeze(0))
        return {
            "best_key_len": None,
            "plaintext":    logits.squeeze(0).argmax(dim=-1),
            "method":       "vanilla",
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=0.01,
            betas=(0.9, 0.98), eps=1e-9,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1, anneal_strategy="cos",
                div_factor=25, final_div_factor=1e4,
            ),
            "interval": "step", "frequency": 1,
        }
        return [optimizer], [scheduler]
