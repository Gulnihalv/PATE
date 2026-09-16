"""FATE - Frequency-Augmented Transformer Encoder.

Like BTE, but prepends a global character-frequency token to the sequence
before the encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.substitution_common.positional_encoding import PositionalEncoding
from models.substitution_common.decoding import ConsistentDecodingMixin


class FreqAugmentedTransformer(ConsistentDecodingMixin, nn.Module):
    def __init__(
        self,
        vocab_size: int = 33,
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
        self.PAD_IDX = 0

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        # MLP that projects the global character frequency into embedding space
        self.freq_encoder = nn.Sequential(
            nn.Linear(vocab_size, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
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

    def compute_global_stats(self, src: torch.Tensor) -> torch.Tensor:
        """Character frequency histogram -> [B, vocab_size] (PAD excluded, normalized)."""
        one_hots = F.one_hot(src, num_classes=self.vocab_size).float()
        mask = (src != self.PAD_IDX).float().unsqueeze(2)
        total_counts = (one_hots * mask).sum(dim=1)
        seq_lengths = mask.sum(dim=1).clamp(min=1)
        return total_counts / seq_lengths

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """src: [B, S] -> logits: [B, S, vocab_size]."""
        B, S = src.shape
        device = src.device
        pad_mask = src == self.PAD_IDX  # [B, S]

        # Frequency token: [B, 1, E]
        global_freqs = self.compute_global_stats(src)
        freq_token = self.freq_encoder(global_freqs).unsqueeze(1)

        # Cipher embedding + positional encoding: [B, S, E]
        src_emb = self.pos_encoding(self.embedding(src))

        # Prepend the frequency token: [B, S+1, E]
        encoder_input = torch.cat([freq_token, src_emb], dim=1)

        freq_pad = torch.zeros(B, 1, dtype=torch.bool, device=device)
        full_pad_mask = torch.cat([freq_pad, pad_mask], dim=1)  # [B, S+1]

        encoder_out = self.encoder(encoder_input, src_key_padding_mask=full_pad_mask)
        token_out = encoder_out[:, 1:, :]  # drop the frequency token -> [B, S, E]

        return self.fc_out(token_out)
