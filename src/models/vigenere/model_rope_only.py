"""RoPE-only encoder (baseline): relative position, no period conditioning.

Identical to model_vanilla except that position enters through rotary encoding
instead of a learned absolute embedding. Still a single forward pass, still no
key-length or validity head.

Isolates what relative position encoding contributes on its own, without the
period ever being resolved explicitly.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# ---------------------------------------------------------------------------- #
#  RoPE                                                                         #
# ---------------------------------------------------------------------------- #
def build_rope_wavelengths(head_dim: int) -> torch.Tensor:
    """Task-agnostic geometric schedule: wavelengths 2 .. 64."""
    n = head_dim // 2
    return 2.0 * (32.0 ** (torch.arange(n).float() / (n - 1)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack((-x1, x0), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = dropout

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, L, _ = x.shape
        H, Dh = self.nhead, self.head_dim
        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)
        q = apply_rope(q, cos[:L], sin[:L])
        k = apply_rope(k, cos[:L], sin[:L])
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = (~key_padding_mask).view(B, 1, 1, L)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, L, H * Dh)
        return self.out_proj(out)


class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = RoPESelfAttention(d_model, nhead, dropout)
        self.linear1   = nn.Linear(d_model, dim_feedforward)
        self.linear2   = nn.Linear(dim_feedforward, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.dropout2  = nn.Dropout(dropout)

    def forward(self, x, cos, sin, key_padding_mask=None):
        h = self.self_attn(x, cos, sin, key_padding_mask)
        x = self.norm1(x + self.dropout1(h))
        h = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(h))
        return x


class VigenereSolver(pl.LightningModule):
    """RoPE encoder, single pass, no period machinery."""

    def __init__(self, vocab_size=29, pad_idx=29, max_len=512,
                 min_key_len=3, max_key_len=12,
                 d_model=256, nhead=8, num_layers=6, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.pad_idx     = pad_idx
        self.min_key_len = min_key_len
        self.max_key_len = max_key_len

        self.char_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)

        head_dim = d_model // nhead
        lam   = build_rope_wavelengths(head_dim)
        theta = 2 * math.pi / lam
        pos   = torch.arange(max_len).float().unsqueeze(1)
        ang   = pos * theta.unsqueeze(0)
        self.register_buffer("rope_cos", torch.repeat_interleave(torch.cos(ang), 2, dim=-1))
        self.register_buffer("rope_sin", torch.repeat_interleave(torch.sin(ang), 2, dim=-1))
        self.register_buffer("rope_wavelengths", lam)

        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, 2048, 0.1) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, key_len=None):
        B, L = src.shape
        pad_mask = (src == self.pad_idx)
        x = self.char_emb(src)
        cos, sin = self.rope_cos.to(x.dtype), self.rope_sin.to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, key_padding_mask=pad_mask)
        return self.fc_out(x)

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
        self.eval()
        logits = self(ciphertext_indices.unsqueeze(0))
        return {
            "best_key_len": None,
            "plaintext":    logits.squeeze(0).argmax(dim=-1),
            "method":       "rope_only",
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
