"""BTE - Base Transformer Encoder.

A plain bidirectional transformer encoder over cipher tokens, without any
frequency information (the no-frequency baseline).
"""

import torch
import torch.nn as nn

from models.substitution_common.positional_encoding import PositionalEncoding
from models.substitution_common.decoding import ConsistentDecodingMixin


class BaseTransformerEncoder(ConsistentDecodingMixin, nn.Module):
    """Embedding + positional encoding -> transformer encoder -> linear head.

    Returns [B, S, vocab_size] logits.
    """

    def __init__(
        self,
        vocab_size: int = 29,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: daha stabil
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        """src: [B, S], pad_mask: [B, S] (True = padding) -> logits: [B, S, vocab_size]."""
        src_emb = self.pos_encoding(self.embedding(src))    # [B, S, E]
        encoder_out = self.encoder(src_emb, src_key_padding_mask=pad_mask)
        return self.fc_out(encoder_out)
