#!/usr/bin/env python3
"""
Loss Functions for P-EAGLE Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def masked_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute masked mean squared error loss.

    Args:
        predictions: [batch, seq_len, hidden_dim]
        targets: [batch, seq_len, hidden_dim]
        mask: [batch, seq_len] - 1 for positions to train on, 0 to ignore

    Returns:
        loss: scalar tensor
    """
    mask_expanded = mask.unsqueeze(-1).float()
    squared_error = (predictions - targets) ** 2
    masked_error = squared_error * mask_expanded

    total_error = masked_error.sum()
    total_tokens = mask.sum()

    if total_tokens > 0:
        loss = total_error / (total_tokens * predictions.shape[-1])
    else:
        # CRITICAL: Empty mask means no learning signal
        import warnings
        warnings.warn("CRITICAL: Loss mask is empty! Model is not learning. Check feature extraction.", RuntimeWarning)
        # Return zero loss that preserves gradient graph
        loss = (predictions * 0).sum()

    return loss


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0
) -> torch.Tensor:
    """
    KL divergence loss for knowledge distillation.

    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        temperature: Softmax temperature

    Returns:
        kl_loss: scalar tensor
    """
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    kl_loss = F.kl_div(
        student_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (temperature ** 2)

    return kl_loss


def hidden_state_token_loss(
    pred_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    target_lm_head: nn.Module,
    mask: torch.Tensor,
    temperature: float = 1.0,
    ce_weight: float = 1.0,
    mse_weight: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss that aligns predicted hidden states with target's token distribution.

    Based on EAGLE paper and official implementation, this uses cross-entropy
    on hard token targets rather than KL divergence on soft distributions.
    Cross-entropy directly optimizes for argmax token matching, which is what
    matters for speculative decoding acceptance.

    P-EAGLE uses the target model's lm_head to convert drafter's predicted
    hidden states to tokens during inference. This loss ensures:
    1. Predicted hidden states produce same tokens as target (via target's lm_head)
    2. Hidden states are close in MSE sense (auxiliary)

    Args:
        pred_hidden: [batch, seq_len, hidden_dim] - drafter predicted hidden states
        target_hidden: [batch, seq_len, hidden_dim] - target model hidden states
        target_lm_head: Target model's lm_head (converts hidden -> logits)
        mask: [batch, seq_len] - 1 for valid positions
        temperature: Softmax temperature (default 1.0, no temperature scaling for CE)
        ce_weight: Weight for cross-entropy loss component (default 1.0)
        mse_weight: Weight for MSE loss component (default 0.1)

    Returns:
        ce_loss: Cross-entropy loss for token prediction
        mse_loss: Mean squared error between hidden states
        accuracy: Token prediction accuracy (%)
    """
    # CRITICAL FIX: Normalize hidden states before lm_head to fix scale mismatch
    # Target hidden states can have extreme values (std=75, max=22528) while
    # drafter predictions are small (std~0.02 from initialization). LayerNorm
    # fixes this by normalizing both to the same scale before lm_head.
    pred_hidden = F.layer_norm(pred_hidden, pred_hidden.shape[-1:])
    target_hidden = F.layer_norm(target_hidden, target_hidden.shape[-1:])

    # Get token distributions from both hidden states using TARGET's lm_head
    # This matches inference: drafter hidden -> target lm_head -> tokens
    pred_logits = target_lm_head(pred_hidden)  # [batch, seq_len, vocab_size]
    target_logits = target_lm_head(target_hidden)

    # Compute HARD targets (argmax tokens) from target distribution
    # Cross-entropy on hard targets directly optimizes for token matching
    target_tokens = target_logits.argmax(dim=-1)  # [batch, seq_len]

    # Flatten for cross-entropy computation
    # Shape: [batch * seq_len, vocab_size] and [batch * seq_len]
    pred_logits_flat = pred_logits.reshape(-1, pred_logits.size(-1))
    target_tokens_flat = target_tokens.reshape(-1)
    mask_flat = mask.reshape(-1)

    # Compute cross-entropy loss per token
    ce_loss_per_token = F.cross_entropy(
        pred_logits_flat,
        target_tokens_flat,
        reduction='none'
    )  # [batch * seq_len]

    # Apply mask and average
    ce_loss = (ce_loss_per_token * mask_flat).sum() / (mask.sum() + 1e-8)
    ce_loss = ce_loss * ce_weight

    # Auxiliary MSE loss for hidden state similarity (helps convergence)
    mse_loss = masked_mse_loss(pred_hidden, target_hidden, mask)
    mse_loss = mse_loss * mse_weight

    # Compute token accuracy (hard metric for monitoring)
    pred_tokens = pred_logits.argmax(dim=-1)
    correct = (pred_tokens == target_tokens).float() * mask
    accuracy = correct.sum() / (mask.sum() + 1e-8) * 100  # percentage

    return ce_loss, mse_loss, accuracy
