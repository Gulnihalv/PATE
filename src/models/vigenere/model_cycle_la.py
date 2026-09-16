"""Per-cycle conditioning with learned absolute positional encoding.

The diagnostic variant of Section 4.5. PATE itself cannot be built with absolute
encoding - it does not converge - so the failure of absolute encoding cannot be
examined there. Supplying more structure than PATE does, through cycle-position
embeddings indexed by (position mod d), makes this variant converge on the
narrower key range, and what it still cannot do is then attributable to the
encoding alone.

Its blind period estimation fails on exactly the key lengths that share no
divisor with the periods its position embeddings develop structure at. The
spectrum of pos_emb and the resulting confusion matrix are produced by
notebooks/vigenere_encoding_analysis.ipynb.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

UNKNOWN_KEY = 0  # sentinel: no key-length conditioning


class VigenereSolver(pl.LightningModule):
    CORRECT_KEY_PROB = 0.40
    WRONG_KEY_PROB   = 0.20
    UNKNOWN_KEY_PROB = 0.40   # = 1 - CORRECT - WRONG

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
        self.pos_emb  = nn.Embedding(max_len, d_model)   # LEARNED ABSOLUTE

        self.unknown_cycle_emb = nn.Parameter(torch.zeros(d_model))

        # A separate cycle-embedding table for each possible key_len
        # cycle_embs[i] -> position table for period key_len = min_key_len + i
        self.cycle_embs = nn.ModuleList([
            nn.Embedding(k, d_model)
            for k in range(min_key_len, max_key_len + 1)
        ])

        # --- Transformer ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Output heads ---
        self.fc_out        = nn.Linear(d_model, vocab_size)      # plaintext
        self.validity_head = nn.Linear(d_model, 1)               # key_len correct?
        self.keylen_head   = nn.Linear(d_model, self.num_key_lens)  # key_len prediction

    def _get_cycle_emb(self, seq_len: int, key_len: torch.Tensor, device) -> torch.Tensor:
        """key_len == UNKNOWN_KEY (0) or any value outside [min,max] -> zeros."""
        B = key_len.size(0)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]

        cycle_emb = torch.zeros(B, seq_len, self.hparams.d_model, device=device)

        unknown_mask = (key_len == UNKNOWN_KEY)
        if unknown_mask.any():
            cycle_emb[unknown_mask] = self.unknown_cycle_emb.view(1, 1, -1)

        for i, k in enumerate(range(self.min_key_len, self.max_key_len + 1)):
            mask = (key_len == k)
            if not mask.any():
                continue
            cycle_pos = positions % k                       # [1, L]
            emb = self.cycle_embs[i](cycle_pos)             # [1, L, d_model]
            cycle_emb[mask] = emb.expand(mask.sum(), -1, -1)

        return cycle_emb

    def forward(self, src: torch.Tensor, key_len: torch.Tensor):
        B, L   = src.shape
        device = src.device

        positions = torch.arange(L, device=device).unsqueeze(0)  # [1, L]
        cycle_emb = self._get_cycle_emb(L, key_len, device)      # [B, L, d_model]

        # Padding mask so the transformer ignores pad tokens
        pad_mask = (src == self.pad_idx)  # [B, L]  True = ignore

        x = self.char_emb(src) + self.pos_emb(positions) + cycle_emb
        encoded = self.transformer(x, src_key_padding_mask=pad_mask)  # [B, L, d_model]

        plaintext_logits = self.fc_out(encoded)                       # [B, L, vocab]

        # Sequence pooling (excluding pad)
        non_pad = (~pad_mask).unsqueeze(-1).float()                   # [B, L, 1]
        pooled  = (encoded * non_pad).sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)  # [B, d_model]

        validity_logit = self.validity_head(pooled).squeeze(-1)       # [B]
        keylen_logits  = self.keylen_head(pooled)                     # [B, num_key_lens]

        return plaintext_logits, validity_logit, keylen_logits

    # ------------------------------------------------------------------ #
    #  Training / validation                                             #
    # ------------------------------------------------------------------ #
    def _sample_modes(self, B, device):
        """Per-sample mode: 0 = CORRECT, 1 = WRONG, 2 = UNKNOWN."""
        r = torch.rand(B, device=device)
        modes = torch.zeros(B, dtype=torch.long, device=device)
        modes[r >= self.CORRECT_KEY_PROB] = 1
        modes[r >= self.CORRECT_KEY_PROB + self.WRONG_KEY_PROB] = 2
        return modes

    def _random_wrong_keys(self, true_key_len):
        """A uniformly random key length guaranteed != true_key_len."""
        B = true_key_len.size(0)
        rand_keys = torch.randint(
            self.min_key_len, self.max_key_len + 1, (B,), device=true_key_len.device
        )
        same = rand_keys == true_key_len
        rand_keys[same] = (rand_keys[same] - self.min_key_len + 1) % self.num_key_lens \
                          + self.min_key_len
        return rand_keys

    def _shared_step(self, batch, use_modes: bool):
        src          = batch["src"]        # [B, L]
        tgt_plain    = batch["tgt_plain"]  # [B, L]
        true_key_len = batch["key_len"]    # [B]

        B      = src.size(0)
        device = src.device

        if use_modes:
            modes = self._sample_modes(B, device)          # [B] 0/1/2
        else:
            # validation: all conditioned on the true key length
            modes = torch.zeros(B, dtype=torch.long, device=device)

        correct_mask = modes == 0
        wrong_mask   = modes == 1
        unknown_mask = modes == 2

        key_len_input = true_key_len.clone()
        if wrong_mask.any():
            key_len_input[wrong_mask] = self._random_wrong_keys(true_key_len)[wrong_mask]
        key_len_input[unknown_mask] = UNKNOWN_KEY

        # Forward
        plaintext_logits, validity_logit, keylen_logits = self(src, key_len_input)

        zero = torch.tensor(0.0, device=device)

        # --- Plaintext CE: only CORRECT-mode samples ---
        if correct_mask.any():
            ce_loss = F.cross_entropy(
                plaintext_logits[correct_mask].reshape(-1, plaintext_logits.size(-1)),
                tgt_plain[correct_mask].reshape(-1),
                ignore_index=self.pad_idx,
            )
        else:
            ce_loss = zero

        # --- Validity BCE: CORRECT (label 1) + WRONG (label 0), UNKNOWN excluded ---
        cond_mask = correct_mask | wrong_mask
        if cond_mask.any():
            validity_label = correct_mask.float()
            bce_loss = F.binary_cross_entropy_with_logits(
                validity_logit[cond_mask], validity_label[cond_mask]
            )
        else:
            bce_loss = zero

        # --- Key-length CE: only UNKNOWN-mode samples ---
        if unknown_mask.any():
            keylen_target = true_key_len[unknown_mask] - self.min_key_len  # 0..num-1
            keylen_loss = F.cross_entropy(keylen_logits[unknown_mask], keylen_target)
        else:
            keylen_loss = zero

        loss = ce_loss + self.VALIDITY_WEIGHT * bce_loss + self.KEYLEN_WEIGHT * keylen_loss

        # --- Metrics ---
        with torch.no_grad():
            # plaintext char accuracy on CORRECT-mode samples
            if correct_mask.any():
                preds     = plaintext_logits[correct_mask].argmax(dim=-1)
                tgt_c     = tgt_plain[correct_mask]
                pmask     = (tgt_c != self.pad_idx)
                acc = ((preds == tgt_c) & pmask).sum().float() / pmask.sum().clamp(min=1)
            else:
                acc = zero

            # validity accuracy on conditioned samples
            if cond_mask.any():
                val_pred = (validity_logit[cond_mask] > 0).float()
                val_acc  = (val_pred == correct_mask.float()[cond_mask]).float().mean()
            else:
                val_acc = zero

            # key-length accuracy on UNKNOWN-mode samples
            if unknown_mask.any():
                k_pred  = keylen_logits[unknown_mask].argmax(dim=-1) + self.min_key_len
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
        # 1) conditioned pass: plaintext quality with the true key length
        loss, ce, bce, _, acc, val_acc, _ = self._shared_step(batch, use_modes=False)

        # 2) blind pass: key-length prediction quality on the whole batch
        src          = batch["src"]
        true_key_len = batch["key_len"]
        unknown = torch.full_like(true_key_len, UNKNOWN_KEY)
        _, _, keylen_logits = self(src, unknown)
        k_pred = keylen_logits.argmax(dim=-1) + self.min_key_len
        keylen_acc = (k_pred == true_key_len).float().mean()

        # hard group: periods coprime to 6 within [3,12] -> {5, 7, 11}
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
    #  Inference                                                          #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def decode(self, ciphertext_indices: torch.Tensor,
               fallback_threshold: float = 0.5,
               use_fallback: bool = True) -> dict:
        """Two-pass decode: predict key length blindly, then decrypt.

        If the validity score of the predicted key length is below
        `fallback_threshold` and `use_fallback` is True, fall back to
        brute-forcing all key lengths (decode_bruteforce).
        """
        self.eval()
        device = ciphertext_indices.device
        src = ciphertext_indices.unsqueeze(0)  # [1, L]

        # Pass 1: blind key-length prediction
        unknown = torch.tensor([UNKNOWN_KEY], device=device)
        _, _, keylen_logits = self(src, unknown)
        keylen_probs = torch.softmax(keylen_logits.squeeze(0), dim=-1)  # [K]
        pred_key_len = int(keylen_probs.argmax().item()) + self.min_key_len

        # Pass 2: conditioned decryption + verification
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
            "plaintext":      plaintext_logits.squeeze(0).argmax(dim=-1),  # [L]
            "validity_score": validity,
            "keylen_probs":   keylen_probs,
            "method":         "predicted",
        }

    @torch.no_grad()
    def decode_bruteforce(self, ciphertext_indices: torch.Tensor) -> dict:
        """Original decode: try every key length in parallel, pick argmax validity."""
        self.eval()
        device = ciphertext_indices.device

        num_keys = self.num_key_lens
        src = ciphertext_indices.unsqueeze(0).expand(num_keys, -1)  # [K, L]
        key_lens = torch.arange(self.min_key_len, self.max_key_len + 1, device=device)

        plaintext_logits, validity_logits, _ = self(src, key_lens)  # [K,L,V], [K]

        validity_scores = torch.sigmoid(validity_logits)  # [K]
        best_idx     = validity_scores.argmax().item()
        best_key_len = self.min_key_len + best_idx

        all_plaintexts = plaintext_logits.argmax(dim=-1)  # [K, L]

        return {
            "best_key_len":    best_key_len,
            "plaintext":       all_plaintexts[best_idx],
            "validity_scores": validity_scores,
            "all_plaintexts":  all_plaintexts,
            "method":          "bruteforce",
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=25,
                final_div_factor=1e4,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
