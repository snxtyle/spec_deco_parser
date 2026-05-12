#!/usr/bin/env python3
"""
Tree Attention for Parallel Eagle (P-EAGLE) Speculative Decoding

Implements tree-structured attention masks for parallel verification
of multiple speculative token sequences.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TreeNode:
    """Node in the speculative tree."""
    token_id: int
    parent: Optional[int] = None  # Index of parent node
    children: List[int] = None  # Indices of child nodes
    depth: int = 0  # Distance from root
    verified: bool = False  # Whether target verified this token

    def __post_init__(self):
        if self.children is None:
            self.children = []


class TreeStructure:
    """
    Manages tree structure for parallel speculative decoding.

    Unlike linear drafting (one chain), tree drafting explores multiple
    paths simultaneously for higher acceptance rates.
    """

    def __init__(self, max_depth: int = 4, branching_factor: int = 2):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.nodes: List[TreeNode] = []
        self.root_children: List[int] = []

    def add_node(self, token_id: int, parent_idx: Optional[int] = None) -> int:
        """Add a node to the tree."""
        depth = 0 if parent_idx is None else self.nodes[parent_idx].depth + 1
        node = TreeNode(token_id=token_id, parent=parent_idx, depth=depth)
        idx = len(self.nodes)
        self.nodes.append(node)

        if parent_idx is not None:
            self.nodes[parent_idx].children.append(idx)
        else:
            self.root_children.append(idx)

        return idx

    def get_path_to_root(self, idx: int) -> List[int]:
        """Get token IDs from root to this node."""
        path = []
        current = idx
        while current is not None:
            path.append(self.nodes[current].token_id)
            current = self.nodes[current].parent
        return list(reversed(path))

    def get_longest_verified_path(self) -> List[int]:
        """Get the longest sequence of verified tokens."""
        max_depth = -1
        best_path = []

        for i, node in enumerate(self.nodes):
            if node.verified and node.depth > max_depth:
                max_depth = node.depth
                best_path = self.get_path_to_root(i)

        return best_path if best_path else []


class TreeAttentionMask:
    """
    Creates attention masks for tree-structured speculative decoding.

    Key innovation: Allows the target model to attend to all speculative
    positions in a single forward pass while maintaining causal structure.
    """

    def __init__(self, speculation_depth: int, branching_factor: int = 1):
        self.K = speculation_depth
        self.branching_factor = branching_factor

    def create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Standard causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return ~mask  # True = can attend, False = cannot attend

    def create_tree_mask(
        self,
        seq_len: int,
        draft_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Create tree attention mask for parallel verification.

        For K speculative tokens, creates mask that allows each position
        to attend to:
        1. All verified context tokens
        2. Previous speculative tokens (causal within speculation)

        Args:
            seq_len: Length of verified sequence
            draft_tokens: [K] speculative token IDs

        Returns:
            mask: [seq_len + K, seq_len + K] boolean mask
        """
        K = draft_tokens.shape[0]
        total_len = seq_len + K

        # Start with causal mask
        mask = self.create_causal_mask(total_len)

        # For speculative positions, ensure they can attend to all verified context
        for i in range(K):
            spec_pos = seq_len + i
            # Can attend to all verified tokens
            mask[spec_pos, :seq_len] = True

        return mask

    def create_position_ids(self, seq_len: int, num_speculative: int) -> torch.Tensor:
        """
        Create position IDs for tree structure.

        For proper RoPE embeddings, speculative tokens need sequential
        position IDs continuing from verified sequence.
        """
        verified_pos = torch.arange(seq_len)
        speculative_pos = torch.arange(seq_len, seq_len + num_speculative)
        return torch.cat([verified_pos, speculative_pos])

    def prepare_tree_inputs(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for tree-based verification.

        Args:
            input_ids: [batch, seq_len] verified tokens
            draft_tokens: [batch, K] speculative tokens
            attention_mask: Optional existing attention mask

        Returns:
            Dictionary with input_ids, attention_mask, position_ids
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        K = draft_tokens.shape[1]
        device = input_ids.device

        # Concatenate verified and draft tokens
        full_input_ids = torch.cat([input_ids, draft_tokens], dim=1)

        # Create tree attention mask
        tree_mask = self.create_tree_mask(seq_len, draft_tokens[0]).to(device)
        tree_mask = tree_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Combine with existing mask if provided
        if attention_mask is not None:
            # Pad existing mask
            padded_mask = F.pad(
                attention_mask,
                (0, K, 0, K),
                value=1  # Allow attention to speculative positions
            )
            # Combine masks
            tree_mask = tree_mask & padded_mask.bool()

        # Create position IDs
        position_ids = self.create_position_ids(seq_len, K).to(device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        return {
            "input_ids": full_input_ids,
            "attention_mask": tree_mask,
            "position_ids": position_ids,
            "verified_len": seq_len,
            "speculative_len": K
        }


def verify_drafts_parallel(
    target_model,
    input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    tree_attention: TreeAttentionMask
) -> Tuple[List[int], int]:
    """
    Verify draft tokens in parallel using tree attention.

    Args:
        target_model: Target language model
        input_ids: [batch, seq_len] current sequence
        draft_tokens: [batch, K] speculative tokens
        tree_attention: TreeAttentionMask instance

    Returns:
        accepted_tokens: List of accepted token IDs
        num_accepted: Number of tokens accepted
    """
    batch_size, seq_len = input_ids.shape
    K = draft_tokens.shape[1]

    # Prepare tree inputs
    inputs = tree_attention.prepare_tree_inputs(input_ids, draft_tokens)

    # Single forward pass for all positions
    with torch.no_grad():
        outputs = target_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"]
        )

    # Get logits for speculative positions only
    logits = outputs.logits[:, seq_len-1:, :]  # [batch, K+1, vocab]

    # Verify each position
    accepted = []
    for i in range(K):
        draft_token = draft_tokens[0, i].item()
        target_token = torch.argmax(logits[0, i]).item()

        if draft_token == target_token:
            accepted.append(draft_token)
        else:
            # Accept target's choice and stop
            accepted.append(target_token)
            break

    return accepted, len(accepted)


def create_speculative_tree(
    drafter,
    input_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    speculation_depth: int,
    branching_factor: int = 1
) -> torch.Tensor:
    """
    Create speculative token tree using drafter.

    For branching_factor > 1, creates multiple candidate paths.
    For now, implements linear chain (branching_factor=1).
    """
    # Get drafter predictions with hidden injection
    with torch.no_grad():
        outputs = drafter.forward(
            input_ids=input_ids,
            target_hidden=target_hidden,
            is_training=False
        )
        mtp_predictions = outputs["mtp_predictions"]

    # Convert hidden states to tokens
    draft_tokens = []
    for k in range(min(speculation_depth, len(mtp_predictions))):
        # Use drafter's lm_head or target's lm_head
        pred_hidden = mtp_predictions[k]
        # Assuming lm_head is available
        logits = pred_hidden  # Placeholder - need actual lm_head
        token = torch.argmax(logits, dim=-1)
        draft_tokens.append(token)

    if draft_tokens:
        return torch.cat(draft_tokens, dim=1)
    else:
        return torch.empty((input_ids.shape[0], 0), dtype=torch.long, device=input_ids.device)
