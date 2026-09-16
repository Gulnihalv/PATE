"""Per-cycle conditioning with rotary encoding (diagnostic variant).

Not the model reported as PATE. Instead of a single period vector, the
candidate key length is supplied through cycle-position embeddings: a separate
table per candidate length, indexed by (position mod d), which gives the
alignment explicitly rather than leaving it to the encoder.

This variant exists because it converges under learned absolute encoding on the
narrower key range (see model_cycle_la), which makes the failure of absolute
encoding observable. This file is its rotary counterpart, used as the reference
point in that comparison. Per-candidate tables do not scale to wide key ranges,
which is why PATE conditions on a single period vector instead.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

UNKNOWN_KEY = 0  # sentinel: no key-length conditioning


# ---------------------------------------------------------------------------- #
#  RoPE                                                                         #
# ---------------------------------------------------------------------------- #
def build_rope_wavelengths(head_dim: int) -> torch.Tensor:
    """Task-agnostic geometric schedule: wavelengths 2 .. 64."""
    n = head_dim // 2
    return 2.0 * (32.0 ** (torch.arange(n).float() / (n - 1)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)   [interleaved pairs]"""
    x = x.view(*x.shape[:-1], -1, 2)
    x0, x1 = x[..., 0], x[..., 1]
    return torch.stack((-x1, x0), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, L, Dh]; cos/sin: [L, Dh] (pairs repeated)."""
    return x * cos + rotate_half(x) * sin


class RoPESelfAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead    = nhead
        self.head_dim = d_model // nhead
        self.scale    = self.head_dim ** -0.5

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout  = dropout

    def forward(self, x, cos, sin, key_padding_mask=None):
        B, L, _ = x.shape
        H, Dh = self.nhead, self.head_dim

        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)   # [B, H, L, Dh]
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        # rotary: q_i^T R_{i-j} k_j  depends only on (i - j)
        q = apply_rope(q, cos[:L], sin[:L])
        k = apply_rope(k, cos[:L], sin[:L])

        attn_mask = None
        if key_padding_mask is not None:
            # True = participate  (SDPA bool-mask convention)
            attn_mask = (~key_padding_mask).view(B, 1, 1, L)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )                                                       # [B, H, L, Dh]
        out = out.transpose(1, 2).reshape(B, L, H * Dh)
        return self.out_proj(out)


class RoPEEncoderLayer(nn.Module):
    """Post-norm layer, mirroring nn.TransformerEncoderLayer defaults."""

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


# ---------------------------------------------------------------------------- #
#  Model                                                                        #
# ---------------------------------------------------------------------------- #
class VigenereSolver(pl.LightningModule):
    CORRECT_KEY_PROB = 0.40
    WRONG_KEY_PROB   = 0.20
    UNKNOWN_KEY_PROB = 0.40

    VALIDITY_WEIGHT = 0.30
    KEYLEN_WEIGHT   = 1.00

    def __init__(
        self,
        vocab_size=29,
        pad_idx=29,
        max_len=512,
        min_key_len=3,
        max_key_len=12,
        d_model=256,
        nhead=8,
        num_layers=6,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.pad_idx      = pad_idx
        self.min_key_len  = min_key_len
        self.max_key_len  = max_key_len
        self.num_key_lens = max_key_len - min_key_len + 1

        self.char_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)

        # --- RoPE tables (no additive positional embedding at all) ---
        head_dim = d_model // nhead
        lam   = build_rope_wavelengths(head_dim)          # [Dh/2]
        theta = 2 * math.pi / lam                                    # [Dh/2]
        pos   = torch.arange(max_len).float().unsqueeze(1)           # [L, 1]
        ang   = pos * theta.unsqueeze(0)                             # [L, Dh/2]
        cos   = torch.repeat_interleave(torch.cos(ang), 2, dim=-1)   # [L, Dh]
        sin   = torch.repeat_interleave(torch.sin(ang), 2, dim=-1)   # [L, Dh]
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
        self.register_buffer("rope_wavelengths", lam)

        self.unknown_cycle_emb = nn.Parameter(torch.zeros(d_model))

        self.cycle_embs = nn.ModuleList([
            nn.Embedding(k, d_model)
            for k in range(min_key_len, max_key_len + 1)
        ])

        self.layers = nn.ModuleList([
            RoPEEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1)
            for _ in range(num_layers)
        ])

        self.fc_out        = nn.Linear(d_model, vocab_size)
        self.validity_head = nn.Linear(d_model, 1)
        self.keylen_head   = nn.Linear(d_model, self.num_key_lens)

    def _get_cycle_emb(self, seq_len: int, key_len: torch.Tensor, device) -> torch.Tensor:
        B = key_len.size(0)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)

        cycle_emb = torch.zeros(B, seq_len, self.hparams.d_model, device=device)

        unknown_mask = (key_len == UNKNOWN_KEY)
        if unknown_mask.any():
            cycle_emb[unknown_mask] = self.unknown_cycle_emb.view(1, 1, -1)

        for i, k in enumerate(range(self.min_key_len, self.max_key_len + 1)):
            mask = (key_len == k)
            if not mask.any():
                continue
            cycle_pos = positions % k
            emb = self.cycle_embs[i](cycle_pos)
            cycle_emb[mask] = emb.expand(mask.sum(), -1, -1)

        return cycle_emb

    def forward(self, src: torch.Tensor, key_len: torch.Tensor):
        B, L   = src.shape
        device = src.device

        cycle_emb = self._get_cycle_emb(L, key_len, device)
        pad_mask  = (src == self.pad_idx)

        # NOTE: no additive positional embedding — position enters via RoPE only.
        x = self.char_emb(src) + cycle_emb

        cos = self.rope_cos.to(x.dtype)
        sin = self.rope_sin.to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, key_padding_mask=pad_mask)
        encoded = x                                                  # [B, L, d_model]

        plaintext_logits = self.fc_out(encoded)

        non_pad = (~pad_mask).unsqueeze(-1).float()
        pooled  = (encoded * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)

        validity_logit = self.validity_head(pooled).squeeze(-1)
        keylen_logits  = self.keylen_head(pooled)

        return plaintext_logits, validity_logit, keylen_logits

    # ------------------------------------------------------------------ #
    #  Training / validation  (unchanged from v2)                        #
    # ------------------------------------------------------------------ #
    def _sample_modes(self, B, device):
        r = torch.rand(B, device=device)
        modes = torch.zeros(B, dtype=torch.long, device=device)
        modes[r >= self.CORRECT_KEY_PROB] = 1
        modes[r >= self.CORRECT_KEY_PROB + self.WRONG_KEY_PROB] = 2
        return modes

    def _random_wrong_keys(self, true_key_len):
        B = true_key_len.size(0)
        rand_keys = torch.randint(
            self.min_key_len, self.max_key_len + 1, (B,), device=true_key_len.device
        )
        same = rand_keys == true_key_len
        rand_keys[same] = (rand_keys[same] - self.min_key_len + 1) % self.num_key_lens \
                          + self.min_key_len
        return rand_keys

    def _shared_step(self, batch, use_modes: bool):
        src          = batch["src"]
        tgt_plain    = batch["tgt_plain"]
        true_key_len = batch["key_len"]

        B      = src.size(0)
        device = src.device

        modes = self._sample_modes(B, device) if use_modes \
            else torch.zeros(B, dtype=torch.long, device=device)

        correct_mask = modes == 0
        wrong_mask   = modes == 1
        unknown_mask = modes == 2

        key_len_input = true_key_len.clone()
        if wrong_mask.any():
            key_len_input[wrong_mask] = self._random_wrong_keys(true_key_len)[wrong_mask]
        key_len_input[unknown_mask] = UNKNOWN_KEY

        plaintext_logits, validity_logit, keylen_logits = self(src, key_len_input)

        zero = torch.tensor(0.0, device=device)

        if correct_mask.any():
            ce_loss = F.cross_entropy(
                plaintext_logits[correct_mask].reshape(-1, plaintext_logits.size(-1)),
                tgt_plain[correct_mask].reshape(-1),
                ignore_index=self.pad_idx,
            )
        else:
            ce_loss = zero

        cond_mask = correct_mask | wrong_mask
        if cond_mask.any():
            validity_label = correct_mask.float()
            bce_loss = F.binary_cross_entropy_with_logits(
                validity_logit[cond_mask], validity_label[cond_mask]
            )
        else:
            bce_loss = zero

        if unknown_mask.any():
            keylen_target = true_key_len[unknown_mask] - self.min_key_len
            keylen_loss = F.cross_entropy(keylen_logits[unknown_mask], keylen_target)
        else:
            keylen_loss = zero

        loss = ce_loss + self.VALIDITY_WEIGHT * bce_loss + self.KEYLEN_WEIGHT * keylen_loss

        with torch.no_grad():
            if correct_mask.any():
                preds = plaintext_logits[correct_mask].argmax(dim=-1)
                tgt_c = tgt_plain[correct_mask]
                pmask = (tgt_c != self.pad_idx)
                acc = ((preds == tgt_c) & pmask).sum().float() / pmask.sum().clamp(min=1)
            else:
                acc = zero

            if cond_mask.any():
                val_pred = (validity_logit[cond_mask] > 0).float()
                val_acc  = (val_pred == correct_mask.float()[cond_mask]).float().mean()
            else:
                val_acc = zero

            if unknown_mask.any():
                k_pred = keylen_logits[unknown_mask].argmax(dim=-1) + self.min_key_len
                keylen_acc = (k_pred == true_key_len[unknown_mask]).float().mean()
            else:
                keylen_acc = zero

        return loss, ce_loss, bce_loss, keylen_loss, acc, val_acc, keylen_acc

    def training_step(self, batch, batch_idx):
        loss, ce, bce, kce, acc, val_acc, keylen_acc = self._shared_step(batch, use_modes=True)
        self.log("train_loss",        loss,       prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_ce_loss",     ce,         prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_bce_loss",    bce,        prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_keylen_loss", kce,        prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc",         acc,        prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_val_acc",     val_acc,    prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_keylen_acc",  keylen_acc, prog_bar=True,  on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ce, bce, _, acc, val_acc, _ = self._shared_step(batch, use_modes=False)

        src          = batch["src"]
        true_key_len = batch["key_len"]
        unknown = torch.full_like(true_key_len, UNKNOWN_KEY)
        _, _, keylen_logits = self(src, unknown)
        k_pred = keylen_logits.argmax(dim=-1) + self.min_key_len
        keylen_acc = (k_pred == true_key_len).float().mean()

        hard = (true_key_len == 5) | (true_key_len == 7) | (true_key_len == 11)
        if hard.any():
            keylen_acc_hard = (k_pred[hard] == true_key_len[hard]).float().mean()
            self.log("val_keylen_acc_hard", keylen_acc_hard,
                     prog_bar=True, on_step=False, on_epoch=True)

        self.log("val_loss",       loss,       prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_ce_loss",    ce,         prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_acc",        acc,        prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_val_acc",    val_acc,    prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_keylen_acc", keylen_acc, prog_bar=True,  on_step=False, on_epoch=True)
        return loss

    # ------------------------------------------------------------------ #
    #  Inference  (unchanged from v2)                                    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode(self, ciphertext_indices, fallback_threshold=0.5, use_fallback=True):
        self.eval()
        device = ciphertext_indices.device
        src = ciphertext_indices.unsqueeze(0)

        unknown = torch.tensor([UNKNOWN_KEY], device=device)
        _, _, keylen_logits = self(src, unknown)
        keylen_probs = torch.softmax(keylen_logits.squeeze(0), dim=-1)
        pred_key_len = int(keylen_probs.argmax().item()) + self.min_key_len

        key_len_t = torch.tensor([pred_key_len], device=device)
        plaintext_logits, validity_logit, _ = self(src, key_len_t)
        validity = torch.sigmoid(validity_logit).item()

        if use_fallback and validity < fallback_threshold:
            result = self.decode_bruteforce(ciphertext_indices)
            result["keylen_probs"] = keylen_probs
            result["method"] = "fallback_bruteforce"
            result["predicted_key_len_pass1"] = pred_key_len
            return result

        return {
            "best_key_len":   pred_key_len,
            "plaintext":      plaintext_logits.squeeze(0).argmax(dim=-1),
            "validity_score": validity,
            "keylen_probs":   keylen_probs,
            "method":         "predicted",
        }

    @torch.no_grad()
    def decode_bruteforce(self, ciphertext_indices):
        self.eval()
        device = ciphertext_indices.device

        num_keys = self.num_key_lens
        src = ciphertext_indices.unsqueeze(0).expand(num_keys, -1)
        key_lens = torch.arange(self.min_key_len, self.max_key_len + 1, device=device)

        plaintext_logits, validity_logits, _ = self(src, key_lens)

        validity_scores = torch.sigmoid(validity_logits)
        best_idx     = validity_scores.argmax().item()
        best_key_len = self.min_key_len + best_idx

        all_plaintexts = plaintext_logits.argmax(dim=-1)

        return {
            "best_key_len":    best_key_len,
            "plaintext":       all_plaintexts[best_idx],
            "validity_scores": validity_scores,
            "all_plaintexts":  all_plaintexts,
            "method":          "bruteforce",
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
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
