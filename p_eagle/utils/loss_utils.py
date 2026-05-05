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
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loss that aligns predicted hidden states with target's token distribution.

    P-EAGLE uses the target model's lm_head to convert drafter's predicted
    hidden states to tokens during inference. This loss ensures:
    1. Predicted hidden states produce same tokens as target (via target's lm_head)
    2. Hidden states are close in MSE sense

    Args:
        pred_hidden: [batch, seq_len, hidden_dim] - drafter predicted hidden states
        target_hidden: [batch, seq_len, hidden_dim] - target model hidden states
        target_lm_head: Target model's lm_head (converts hidden -> logits)
        mask: [batch, seq_len] - 1 for valid positions
        temperature: Softmax temperature

    Returns:
        token_loss: Cross-entropy between predicted and target token distributions
        mse_loss: Mean squared error between hidden states
        accuracy: Token prediction accuracy (%)
    """
    print(f"DEBUG: hidden_state_token_loss called, mask_sum={mask.sum().item()}")
    # Get token distributions from both hidden states using TARGET's lm_head
    # This matches inference: drafter hidden -> target lm_head -> tokens
    pred_logits = target_lm_head(pred_hidden)  # [batch, seq_len, vocab_size]
    target_logits = target_lm_head(target_hidden)

    # Compute KL divergence between token distributions
    # This is softer than hard accuracy - encourages similar probability distributions
    pred_log_probs = F.log_softmax(pred_logits / temperature, dim=-1)
    target_probs = F.softmax(target_logits / temperature, dim=-1)

    # KL(P_target || P_pred) - measures how much distributions differ
    kl_loss = F.kl_div(
        pred_log_probs,
        target_probs,
        reduction='none'
    ).sum(dim=-1)  # [batch, seq_len]

    # Apply mask and average (divide by vocab size to normalize)
    vocab_size = pred_logits.shape[-1]
    masked_kl = (kl_loss * mask).sum() / (mask.sum() * vocab_size + 1e-8)

    # DEBUG: Print loss components
    if mask.sum() > 0:
        raw_kl = kl_loss.sum().item()
        print(f"  DEBUG KL: raw={raw_kl:.2f}, masked={masked_kl.item():.6f}, vocab={vocab_size}, tokens={mask.sum().item()}")

    # Also compute MSE for hidden state similarity
    mse_loss = masked_mse_loss(pred_hidden, target_hidden, mask)

    # Compute token accuracy (hard metric for monitoring)
    pred_tokens = pred_logits.argmax(dim=-1)
    target_tokens = target_logits.argmax(dim=-1)
    correct = (pred_tokens == target_tokens).float() * mask
    accuracy = correct.sum() / (mask.sum() + 1e-8) * 100  # percentage

    return masked_kl, mse_loss, accuracy
