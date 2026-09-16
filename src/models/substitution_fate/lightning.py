"""FATE LightningModule. Wraps FreqAugmentedTransformer with the shared
training/validation/optimizer logic from SubstitutionLightningBase.
"""

import torch.nn as nn
import torchmetrics

from models.substitution_common.lightning_base import SubstitutionLightningBase
from models.substitution_fate.model import FreqAugmentedTransformer


class SubstitutionCipherSolverFreq(SubstitutionLightningBase):
    def __init__(
        self,
        vocab_size: int = 33,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        warmup_steps: int = 400,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FreqAugmentedTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_len=max_len,
        )

        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=label_smoothing
        )
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=vocab_size, ignore_index=0
        )
