"""Shared LightningModule base for the substitution transformer models (BTE, FATE).

Holds the common training/validation steps and the Noam-scheduled optimizer.
Subclasses set self.model (the encoder), self.loss_fn and self.accuracy in
their __init__ and call self.save_hyperparameters().
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


class SubstitutionLightningBase(pl.LightningModule):

    def forward(self, src):
        return self.model(src)

    def training_step(self, batch, batch_idx):
        src, _tgt_input, tgt_output = batch
        logits = self.model(src)               # [B, S, V]
        loss = self.loss_fn(
            logits.reshape(-1, self.hparams.vocab_size),
            tgt_output.reshape(-1),
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        src, _tgt_input, tgt_output = batch
        logits = self.model(src)
        loss = self.loss_fn(
            logits.reshape(-1, self.hparams.vocab_size),
            tgt_output.reshape(-1),
        )
        preds = logits.argmax(dim=-1)
        acc = self.accuracy(preds.reshape(-1), tgt_output.reshape(-1))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc",  acc,  prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # Noam (Transformer) learning-rate schedule; AdamW with lr=1.0 lets the
        # scheduler control the effective learning rate.
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-2,
        )
        d_model = self.hparams.embed_dim
        warmup = self.hparams.warmup_steps

        def noam_lambda(step: int) -> float:
            step = max(1, step)
            return (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
