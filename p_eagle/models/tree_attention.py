#!/usr/bin/env python3
"""
Tree Attention Mask for Parallel Speculative Decoding Verification
"""

import torch
from typing import Tuple


class TreeAttentionMask:
    """
    Implements tree-style attention mask for parallel verification.
    For K speculative tokens, creates a causal mask that allows the target
    model to verify all positions in a single forward pass.
    """

    def __init__(self, speculation_depth: int):
        self.K = speculation_depth

    def create_mask(self, seq_len: int, num_speculative: int) -> torch.Tensor:
        """
        Create tree attention mask for speculative verification.

        Args:
            seq_len: Current sequence length
            num_speculative: Number of speculative tokens added

        Returns:
            mask: [seq_len + num_speculative, seq_len + num_speculative]
        """
        total_len = seq_len + num_speculative
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)

        # Standard causal mask for verified portion
        for i in range(seq_len):
            mask[i, :i+1] = True

        # Tree connections for speculative tokens
        for i in range(num_speculative):
            pos = seq_len + i
            # Attend to all verified tokens
            mask[pos, :seq_len] = True
            # Attend to previous speculative tokens in chain
            mask[pos, seq_len:pos+1] = True

        return mask

    def create_position_ids(self, seq_len: int, num_speculative: int) -> torch.Tensor:
        """
        Create position IDs for tree structure.

        Args:
            seq_len: Current sequence length
            num_speculative: Number of speculative tokens

        Returns:
            position_ids: [seq_len + num_speculative]
        """
        verified_pos = torch.arange(seq_len)
        speculative_pos = torch.full((num_speculative,), seq_len - 1)
        return torch.cat([verified_pos, speculative_pos])

    def create_tree_inputs(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create full tree inputs for parallel verification.

        Args:
            input_ids: [batch, seq_len] verified tokens
            draft_tokens: [batch, k] speculative tokens

        Returns:
            full_input_ids: [batch, seq_len + k]
            attention_mask: [batch, seq_len + k, seq_len + k]
            position_ids: [batch, seq_len + k]
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        k = draft_tokens.shape[1]

        # Concatenate inputs
        full_input_ids = torch.cat([input_ids, draft_tokens], dim=1)

        # Create tree attention mask
        mask = self.create_mask(seq_len, k)
        attention_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Create position IDs
        position_ids = self.create_position_ids(seq_len, k)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        return full_input_ids, attention_mask, position_ids
