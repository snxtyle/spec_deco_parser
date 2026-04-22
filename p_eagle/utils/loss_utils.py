#!/usr/bin/env python3
"""
Loss Functions for P-EAGLE Training
"""

import torch
import torch.nn.functional as F


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
        loss = torch.tensor(0.0, device=predictions.device)

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
