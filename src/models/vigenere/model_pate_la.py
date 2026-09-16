import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

UNKNOWN_KEY = 0  # sentinel: no key-length conditioning


class VigenereSolver(pl.LightningModule):
    CORRECT_KEY_PROB = 0.40
    WRONG_KEY_PROB   = 0.20
    UNKNOWN_KEY_PROB = 0.40

    VALIDITY_WEIGHT = 0.30
    KEYLEN_WEIGHT   = 1.0

    def __init__(self, vocab_size=29, pad_idx=29, max_len=512,
                 min_key_len=3, max_key_len=12,
                 d_model=256, nhead=8, num_layers=6, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.pad_idx      = pad_idx
        self.min_key_len  = min_key_len
        self.max_key_len  = max_key_len
        self.num_key_lens = max_key_len - min_key_len + 1

        self.char_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=pad_idx)
        self.pos_emb  = nn.Embedding(max_len, d_model)   # LEARNED ABSOLUTE

        self.period_emb = nn.Embedding(self.num_key_lens + 1, d_model)
        nn.init.zeros_(self.period_emb.weight[0])   # UNKNOWN starts neutral

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out        = nn.Linear(d_model, vocab_size)
        self.validity_head = nn.Linear(d_model, 1)
        self.keylen_head   = nn.Linear(d_model, self.num_key_lens)

    def _period_vector(self, key_len: torch.Tensor) -> torch.Tensor:
        """key_len -> [B, 1, d_model], broadcast over positions (no mod-d index)."""
        idx = torch.where(
            (key_len >= self.min_key_len) & (key_len <= self.max_key_len),
            key_len - self.min_key_len + 1,
            torch.zeros_like(key_len),
        )
        return self.period_emb(idx).unsqueeze(1)

    def forward(self, src, key_len):
        B, L = src.shape
        device = src.device
        pad_mask = (src == self.pad_idx)
        positions = torch.arange(L, device=device).unsqueeze(0)

        x = self.char_emb(src) + self.pos_emb(positions) + self._period_vector(key_len)   # uniform, not per-cycle
        encoded = self.transformer(x, src_key_padding_mask=pad_mask)

        plaintext_logits = self.fc_out(encoded)
        non_pad = (~pad_mask).unsqueeze(-1).float()
        pooled  = (encoded * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)
        return plaintext_logits, self.validity_head(pooled).squeeze(-1), self.keylen_head(pooled)

    # ---- training / validation: identical to PATE ----
    def _sample_modes(self, B, device):
        r = torch.rand(B, device=device)
        modes = torch.zeros(B, dtype=torch.long, device=device)
        modes[r >= self.CORRECT_KEY_PROB] = 1
        modes[r >= self.CORRECT_KEY_PROB + self.WRONG_KEY_PROB] = 2
        return modes

    def _random_wrong_keys(self, true_key_len):
        B = true_key_len.size(0)
        rand_keys = torch.randint(self.min_key_len, self.max_key_len + 1,
                                  (B,), device=true_key_len.device)
        same = rand_keys == true_key_len
        rand_keys[same] = (rand_keys[same] - self.min_key_len + 1) % self.num_key_lens \
                          + self.min_key_len
        return rand_keys

    def _shared_step(self, batch, use_modes: bool):
        src, tgt_plain, true_key_len = batch["src"], batch["tgt_plain"], batch["key_len"]
        B, device = src.size(0), src.device

        modes = self._sample_modes(B, device) if use_modes \
            else torch.zeros(B, dtype=torch.long, device=device)
        correct_mask, wrong_mask, unknown_mask = modes == 0, modes == 1, modes == 2

        key_len_input = true_key_len.clone()
        if wrong_mask.any():
            key_len_input[wrong_mask] = self._random_wrong_keys(true_key_len)[wrong_mask]
        key_len_input[unknown_mask] = UNKNOWN_KEY

        plaintext_logits, validity_logit, keylen_logits = self(src, key_len_input)
        zero = torch.tensor(0.0, device=device)

        if correct_mask.any():
            ce_loss = F.cross_entropy(
                plaintext_logits[correct_mask].reshape(-1, plaintext_logits.size(-1)),
                tgt_plain[correct_mask].reshape(-1), ignore_index=self.pad_idx)
        else:
            ce_loss = zero

        cond_mask = correct_mask | wrong_mask
        if cond_mask.any():
            bce_loss = F.binary_cross_entropy_with_logits(
                validity_logit[cond_mask], correct_mask.float()[cond_mask])
        else:
            bce_loss = zero

        if unknown_mask.any():
            keylen_loss = F.cross_entropy(
                keylen_logits[unknown_mask], true_key_len[unknown_mask] - self.min_key_len)
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
                val_acc = ((validity_logit[cond_mask] > 0).float()
                           == correct_mask.float()[cond_mask]).float().mean()
            else:
                val_acc = zero
            if unknown_mask.any():
                k_pred = keylen_logits[unknown_mask].argmax(dim=-1) + self.min_key_len
                keylen_acc = (k_pred == true_key_len[unknown_mask]).float().mean()
            else:
                keylen_acc = zero

        return loss, ce_loss, bce_loss, keylen_loss, acc, val_acc, keylen_acc

    def training_step(self, batch, batch_idx):
        loss, ce, bce, kce, acc, val_acc, keylen_acc = self._shared_step(batch, True)
        self.log("train_loss",       loss,       prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_acc",        acc,        prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_keylen_acc", keylen_acc, prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_ce_loss",    ce,         prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_bce_loss",   bce,        prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_keylen_loss",kce,        prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, ce, bce, _, acc, val_acc, _ = self._shared_step(batch, False)

        src, true_key_len = batch["src"], batch["key_len"]
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
        self.log("val_acc",        acc,        prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_keylen_acc", keylen_acc, prog_bar=True,  on_step=False, on_epoch=True)
        self.log("val_val_acc",    val_acc,    prog_bar=False, on_step=False, on_epoch=True)
        self.log("period_emb_norm", self.period_emb.weight[1:].norm(), prog_bar=False, on_step=False, on_epoch=True)
        return loss

    # ---- inference: two passes, as in PATE ----
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
            result.update(keylen_probs=keylen_probs, method="fallback_bruteforce",
                          predicted_key_len_pass1=pred_key_len)
            return result

        return {"best_key_len": pred_key_len,
                "plaintext": plaintext_logits.squeeze(0).argmax(dim=-1),
                "validity_score": validity, "keylen_probs": keylen_probs,
                "method": "predicted"}

    @torch.no_grad()
    def decode_bruteforce(self, ciphertext_indices):
        self.eval()
        device = ciphertext_indices.device
        src = ciphertext_indices.unsqueeze(0).expand(self.num_key_lens, -1)
        key_lens = torch.arange(self.min_key_len, self.max_key_len + 1, device=device)
        plaintext_logits, validity_logits, _ = self(src, key_lens)
        validity_scores = torch.sigmoid(validity_logits)
        best_idx = validity_scores.argmax().item()
        all_plaintexts = plaintext_logits.argmax(dim=-1)
        return {"best_key_len": self.min_key_len + best_idx,
                "plaintext": all_plaintexts[best_idx],
                "validity_scores": validity_scores,
                "all_plaintexts": all_plaintexts, "method": "bruteforce"}

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
