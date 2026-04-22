#!/usr/bin/env python3
"""
P-EAGLE Drafter Training Script

Trains a Drafter model to predict K future hidden states of the Target Model
using Multi-Token Prediction (MTP) heads with parallel speculation.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from bitsandbytes.optim import PagedAdamW8bit
from tqdm import tqdm
import numpy as np

from ..models.eagle_drafter import EagleDrafterModel
from ..utils.feature_utils import EagleTrainingDataset
from ..utils.loss_utils import masked_mse_loss
from ..utils.metrics import MetricsTracker, GenerationMetrics


class EagleTrainer:
    """Trainer for P-EAGLE Drafter model."""

    def __init__(
        self,
        drafter_model_name: str,
        target_hidden_dim: int,
        feature_dir: str,
        output_dir: str,
        speculation_depth: int = 4,
        use_lora: bool = True,
        lora_rank: int = 64,
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        save_every: int = 1000,
        device: str = "cuda"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every
        self.device = device
        self.global_step = 0

        # Initialize model
        self.model = EagleDrafterModel(
            base_model_name=drafter_model_name,
            target_hidden_dim=target_hidden_dim,
            speculation_depth=speculation_depth,
            use_lora=use_lora,
            lora_rank=lora_rank,
            device=device
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(drafter_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Setup optimizer
        print("Setting up PagedAdamW8bit optimizer...")
        params_with_wd = []
        params_without_wd = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name:
                    params_without_wd.append(param)
                else:
                    params_with_wd.append(param)

        self.optimizer = PagedAdamW8bit(
            [
                {"params": params_with_wd, "weight_decay": 0.01},
                {"params": params_without_wd, "weight_decay": 0.0}
            ],
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Setup dataset
        self.dataset = EagleTrainingDataset(
            feature_dir=feature_dir,
            tokenizer=self.tokenizer,
            speculation_depth=speculation_depth
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0
        )

        # Setup scheduler
        total_steps = len(self.dataloader) * num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")
        self.metrics_tracker = MetricsTracker()

        print(f"Training setup complete:")
        print(f"  Samples: {len(self.dataset)}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total steps: {total_steps}")

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for batching."""
        input_ids = nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        target_hidden = nn.utils.rnn.pad_sequence(
            [b["target_hidden"] for b in batch],
            batch_first=True,
            padding_value=0.0
        )
        loss_mask = nn.utils.rnn.pad_sequence(
            [b["loss_mask"] for b in batch],
            batch_first=True,
            padding_value=0
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0
        )

        return {
            "input_ids": input_ids.to(self.device),
            "target_hidden": target_hidden.to(self.device),
            "loss_mask": loss_mask.to(self.device),
            "attention_mask": attention_mask.to(self.device)
        }

    def train(self):
        """Main training loop."""
        self.model.train()
        best_loss = float("inf")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            epoch_losses = []
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(pbar):
                loss, metrics = self._training_step(batch)

                epoch_losses.append(loss.item())
                self.global_step += 1

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                # Log to TensorBoard
                if self.global_step % 10 == 0:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                    for k, v in metrics.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)

                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self._save_checkpoint(f"checkpoint_step_{self.global_step}")

            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint("best_model")
                print(f"New best model saved!")

        self.writer.close()
        print("\nTraining complete!")

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Single training step."""
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        # Compute losses for each MTP head
        total_loss = 0.0
        mtp_losses = []

        for k, pred_hidden in enumerate(outputs["mtp_predictions"]):
            shift = k + 1

            if pred_hidden.shape[1] > 0:
                target_shifted = batch["target_hidden"][:, shift:shift + pred_hidden.shape[1]]
                mask_shifted = batch["loss_mask"][:, shift:shift + pred_hidden.shape[1]]

                loss_k = masked_mse_loss(pred_hidden, target_shifted, mask_shifted)
                mtp_losses.append(loss_k.item() if loss_k.item() > 0 else 0.0)
                total_loss += loss_k

        if len(outputs["mtp_predictions"]) > 0:
            total_loss = total_loss / len(outputs["mtp_predictions"])

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        metrics = {
            "mtp_loss_avg": np.mean(mtp_losses) if mtp_losses else 0.0,
        }
        for i, loss_i in enumerate(mtp_losses):
            metrics[f"mtp_loss_{i+1}"] = loss_i

        return total_loss, metrics

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        self.model.save_checkpoint(str(checkpoint_dir))
        print(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Drafter Training")
    parser.add_argument("--drafter_model", required=True, help="Base model for drafter")
    parser.add_argument("--target_hidden_dim", type=int, required=True)
    parser.add_argument("--speculation_depth", type=int, default=4)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)

    args = parser.parse_args()

    trainer = EagleTrainer(
        drafter_model_name=args.drafter_model,
        target_hidden_dim=args.target_hidden_dim,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        speculation_depth=args.speculation_depth,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps
    )

    trainer.train()


if __name__ == "__main__":
    main()
