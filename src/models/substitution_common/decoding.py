"""Consistent decoding mixin.

Turns the raw per-position argmax into a bijective cipher->plain mapping by
majority vote, then applies that mapping. Attach to any nn.Module whose
forward(src) returns [B, S, vocab] logits.
"""

import torch


class ConsistentDecodingMixin:

    @torch.no_grad()
    def generate(self, src: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        logits = self.forward(src, pad_mask)          # [B, S, V]
        return logits.argmax(dim=-1)         # [B, S]

    @torch.no_grad()
    def generate_consistent(self, src: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
        raw_prediction = self.generate(src, pad_mask)

        results = []
        for b in range(src.size(0)):
            cipher_seq = src[b].tolist()
            pred_seq = raw_prediction[b].tolist()

            votes: dict[int, dict[int, int]] = {}
            pad_seq = pad_mask[b].tolist() if pad_mask is not None else [False] * len(cipher_seq)
            for c_char, p_char, is_pad in zip(cipher_seq, pred_seq, pad_seq):
                if is_pad:
                    continue
                votes.setdefault(c_char, {}).setdefault(p_char, 0)
                votes[c_char][p_char] += 1

            final_key_map: dict[int, int] = {}
            used_plain_chars: set[int] = set()

            sorted_cipher_chars = sorted(
                votes.keys(), key=lambda k: sum(votes[k].values()), reverse=True
            )
            for c_char in sorted_cipher_chars:
                sorted_cands = sorted(
                    votes[c_char].items(), key=lambda x: x[1], reverse=True
                )
                best = sorted_cands[0][0]
                for cand, _ in sorted_cands:
                    if cand not in used_plain_chars:
                        best = cand
                        break
                final_key_map[c_char] = best
                used_plain_chars.add(best)

            new_seq = [
                final_key_map.get(c, pred_seq[i]) if not pad_seq[i] else c
                for i, c in enumerate(cipher_seq)
            ]
            results.append(torch.tensor(new_seq, device=src.device))

        return torch.stack(results)
