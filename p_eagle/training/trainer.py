#!/usr/bin/env python3
"""
P-EAGLE Drafter Training Script

Trains a Drafter model to predict K future hidden states of the Target Model
using Multi-Token Prediction (MTP) heads with parallel speculation.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import timedelta

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


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information."""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": []
    }

    if not torch.cuda.is_available():
        return info

    info["device_count"] = torch.cuda.device_count()

    for i in range(info["device_count"]):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)  # GB

        # Get current memory usage
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        free = total_memory - allocated

        info["devices"].append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(total_memory, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
            "multi_processor_count": props.multi_processor_count,
            "compute_capability": f"{props.major}.{props.minor}"
        })

    return info


def estimate_vram_requirements(
    drafter_params_b: float,
    target_hidden_dim: int,
    batch_size: int,
    seq_length: int = 2048,
    speculation_depth: int = 4,
    use_lora: bool = True
) -> Dict[str, float]:
    """
    Estimate VRAM requirements in GB.

    Args:
        drafter_params_b: Drafter model size in billions (e.g., 1.5 for 1.5B)
        target_hidden_dim: Target model hidden dimension
        batch_size: Training batch size
        seq_length: Maximum sequence length
        speculation_depth: Number of MTP heads
        use_lora: Whether using LoRA
    """
    # Base model memory (parameters in fp16/bf16)
    param_bytes = 2 if use_lora else 4  # LoRA keeps base frozen in 16-bit
    base_model_gb = drafter_params_b * param_bytes

    # LoRA parameters (if enabled) - typically ~0.5-2% of base
    lora_gb = base_model_gb * 0.01 if use_lora else 0

    # Gradients (only for trainable params)
    gradients_gb = lora_gb if use_lora else base_model_gb

    # Optimizer states (PagedAdamW8bit uses 8-bit, but let's be conservative)
    optimizer_gb = lora_gb * 2 if use_lora else base_model_gb * 2

    # Activations (forward pass)
    # Rough estimate: batch * seq * hidden * layers * 4 bytes
    # Assuming ~24 layers average, 2x buffer for intermediate activations
    est_layers = 24 if drafter_params_b >= 1.5 else 16
    # Use average hidden dim: 2048 covers 1536-4096 range (Qwen to Llama)
    avg_hidden_dim = 2048
    activation_gb = (batch_size * seq_length * avg_hidden_dim * est_layers * 4) / (1024**3)

    # MTP heads memory (parallel predictions)
    mtp_head_gb = (speculation_depth * batch_size * seq_length * target_hidden_dim * 4) / (1024**3)

    # Feature cache during training
    feature_cache_gb = (batch_size * seq_length * target_hidden_dim * 4) / (1024**3)

    # System overhead (CUDA context, fragmentation, etc.)
    overhead_gb = 2.0

    total_gb = (
        base_model_gb +
        lora_gb +
        gradients_gb +
        optimizer_gb +
        activation_gb +
        mtp_head_gb +
        feature_cache_gb +
        overhead_gb
    )

    return {
        "base_model_gb": round(base_model_gb, 2),
        "lora_params_gb": round(lora_gb, 2),
        "gradients_gb": round(gradients_gb, 2),
        "optimizer_states_gb": round(optimizer_gb, 2),
        "activations_gb": round(activation_gb, 2),
        "mtp_heads_gb": round(mtp_head_gb, 2),
        "feature_cache_gb": round(feature_cache_gb, 2),
        "overhead_gb": overhead_gb,
        "total_required_gb": round(total_gb, 2),
        "recommended_gb": round(total_gb * 1.2, 2)  # 20% safety margin
    }


def estimate_training_time(
    num_samples: int,
    num_epochs: int,
    batch_size: int,
    steps_per_sec: float = 1.5
) -> Dict[str, Any]:
    """Estimate total training time."""
    steps_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_steps = steps_per_epoch * num_epochs
    total_seconds = total_steps / steps_per_sec

    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_total_seconds": total_seconds,
        "estimated_total_time": str(timedelta(seconds=int(total_seconds))),
        "time_per_epoch": str(timedelta(seconds=int(steps_per_epoch / steps_per_sec)))
    }


def check_disk_space(path: str, required_gb: float = 10.0) -> Dict[str, Any]:
    """Check available disk space."""
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    total_gb = usage.total / (1024**3)

    return {
        "path": path,
        "total_gb": round(total_gb, 2),
        "free_gb": round(free_gb, 2),
        "required_gb": required_gb,
        "sufficient": free_gb >= required_gb
    }


def parse_model_size(model_name: str) -> float:
    """Extract model size in billions from name."""
    import re
    # Look for patterns like 1.5B, 7B, 0.5B, etc.
    match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
    if match:
        return float(match.group(1))
    # Check for common sizes in name
    if "0.5" in model_name.lower():
        return 0.5
    elif "1.5" in model_name.lower():
        return 1.5
    elif "3" in model_name.lower():
        return 3.0
    elif "7" in model_name.lower():
        return 7.0
    return 1.5  # Default assumption


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
        device: str = "cuda",
        skip_hardware_check: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir = feature_dir
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every
        self.device = device
        self.global_step = 0
        self.drafter_model_name = drafter_model_name

        # Hardware requirement check
        if not skip_hardware_check:
            self._run_hardware_check(
                drafter_model_name=drafter_model_name,
                target_hidden_dim=target_hidden_dim,
                batch_size=batch_size,
                speculation_depth=speculation_depth,
                use_lora=use_lora,
                num_epochs=num_epochs
            )

        # Initialize model
        self.model = EagleDrafterModel(
            base_model_name=drafter_model_name,
            target_hidden_dim=target_hidden_dim,
            speculation_depth=speculation_depth,
            use_lora=use_lora,
            lora_rank=lora_rank,
            device=device
        )

        # Load tokenizer (use same cache as model)
        import os
        cache_dir = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(drafter_model_name, cache_dir=cache_dir)
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

    def _run_hardware_check(
        self,
        drafter_model_name: str,
        target_hidden_dim: int,
        batch_size: int,
        speculation_depth: int,
        use_lora: bool,
        num_epochs: int
    ):
        """Run comprehensive hardware requirement check before training."""
        print_section("P-EAGLE TRAINING - HARDWARE CHECK")

        # Parse model size
        model_size_b = parse_model_size(drafter_model_name)
        print(f"\n📊 Model: {drafter_model_name}")
        print(f"   Estimated size: ~{model_size_b}B parameters")
        print(f"   Target hidden dim: {target_hidden_dim}")
        print(f"   Speculation depth (K): {speculation_depth}")
        print(f"   LoRA enabled: {use_lora}")

        # GPU Check
        print_section("GPU AVAILABILITY")
        gpu_info = get_gpu_info()

        if not gpu_info["cuda_available"]:
            print("❌ ERROR: CUDA not available!")
            print("   Training requires at least one NVIDIA GPU.")
            raise RuntimeError("CUDA not available. GPU is required for training.")

        print(f"✅ CUDA available")
        print(f"   Device count: {gpu_info['device_count']}")

        total_gpu_memory = 0
        for dev in gpu_info["devices"]:
            print(f"\n   GPU {dev['index']}: {dev['name']}")
            print(f"   ├── Total memory: {dev['total_memory_gb']:.2f} GB")
            print(f"   ├── Free memory: {dev['free_gb']:.2f} GB")
            print(f"   ├── Compute capability: {dev['compute_capability']}")
            print(f"   └── Multi-processors: {dev['multi_processor_count']}")
            total_gpu_memory += dev['total_memory_gb']

        # VRAM Estimation
        print_section("VRAM REQUIREMENTS")
        vram_req = estimate_vram_requirements(
            drafter_params_b=model_size_b,
            target_hidden_dim=target_hidden_dim,
            batch_size=batch_size,
            speculation_depth=speculation_depth,
            use_lora=use_lora
        )

        print(f"\nEstimated VRAM breakdown:")
        print(f"  Base model ({'16-bit' if use_lora else '32-bit'}): {vram_req['base_model_gb']:.2f} GB")
        if use_lora:
            print(f"  LoRA parameters: {vram_req['lora_params_gb']:.2f} GB")
            print(f"  Gradients (LoRA only): {vram_req['gradients_gb']:.2f} GB")
        else:
            print(f"  Gradients (full): {vram_req['gradients_gb']:.2f} GB")
        print(f"  Optimizer states: {vram_req['optimizer_states_gb']:.2f} GB")
        print(f"  Activations: {vram_req['activations_gb']:.2f} GB")
        print(f"  MTP heads: {vram_req['mtp_heads_gb']:.2f} GB")
        print(f"  Feature cache: {vram_req['feature_cache_gb']:.2f} GB")
        print(f"  System overhead: {vram_req['overhead_gb']:.2f} GB")
        print(f"\n{'─'*50}")
        print(f"  TOTAL REQUIRED: ~{vram_req['total_required_gb']:.2f} GB")
        print(f"  RECOMMENDED: ~{vram_req['recommended_gb']:.2f} GB (20% margin)")

        # Check if sufficient VRAM
        if vram_req['recommended_gb'] > total_gpu_memory:
            print(f"\n⚠️  WARNING: Insufficient GPU memory!")
            print(f"   You have: {total_gpu_memory:.2f} GB total")
            print(f"   Recommended: {vram_req['recommended_gb']:.2f} GB")
            print(f"\n   Suggestions:")
            print(f"   • Reduce batch_size (currently {batch_size})")
            print(f"   • Use a smaller drafter model")
            print(f"   • Reduce speculation_depth (currently {speculation_depth})")
            print(f"   • Use gradient accumulation instead")

            user_input = input("\nContinue anyway? [y/N]: ").strip().lower()
            if user_input != 'y':
                raise RuntimeError("Hardware requirements not met. Aborting.")
        else:
            print(f"\n✅ Sufficient GPU memory available")

        # Disk Space Check
        print_section("DISK SPACE CHECK")
        # Estimate: base model (~3GB) + checkpoints (~10GB) + logs (~1GB)
        estimated_need_gb = 15 + (model_size_b * 2)
        disk_check = check_disk_space(str(self.output_dir), estimated_need_gb)

        print(f"Output directory: {disk_check['path']}")
        print(f"  Total: {disk_check['total_gb']:.2f} GB")
        print(f"  Free: {disk_check['free_gb']:.2f} GB")
        print(f"  Estimated need: ~{estimated_need_gb:.2f} GB")

        if not disk_check['sufficient']:
            print(f"\n⚠️  WARNING: Low disk space!")
            user_input = input("Continue anyway? [y/N]: ").strip().lower()
            if user_input != 'y':
                raise RuntimeError("Insufficient disk space. Aborting.")
        else:
            print(f"✅ Sufficient disk space")

        # Training Time Estimation
        print_section("TRAINING TIME ESTIMATE")

        # Count feature files
        feature_files = list(Path(self.feature_dir if hasattr(self, 'feature_dir') else ".").glob("*.pt"))
        num_samples = len(feature_files)

        if num_samples > 0:
            time_est = estimate_training_time(
                num_samples=num_samples,
                num_epochs=num_epochs,
                batch_size=batch_size
            )

            print(f"Dataset: {num_samples} samples")
            print(f"Epochs: {num_epochs}")
            print(f"Steps per epoch: {time_est['steps_per_epoch']}")
            print(f"Total steps: {time_est['total_steps']}")
            print(f"\n⏱️  Estimated training time:")
            print(f"   Per epoch: {time_est['time_per_epoch']}")
            print(f"   Total: ~{time_est['estimated_total_time']}")
        else:
            print(f"⚠️  No feature files found yet. Cannot estimate training time.")

        # Model Download Info
        print_section("MODEL DOWNLOAD INFO")
        print(f"Drafter model will be downloaded from HuggingFace:")
        print(f"  {drafter_model_name}")
        print(f"\nCache location: ~/.cache/huggingface/hub/")
        print(f"Download size: ~{model_size_b * 2:.1f} GB (weights + tokenizer)")

        # Final confirmation
        print_section("READY TO START")
        print("\nConfiguration summary:")
        print(f"  Drafter: {drafter_model_name}")
        print(f"  Target hidden dim: {target_hidden_dim}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  LoRA rank: {64 if use_lora else 'N/A (full fine-tune)'}")
        print(f"  Output dir: {self.output_dir}")

        user_input = input("\nStart training? [Y/n]: ").strip().lower()
        if user_input == 'n':
            raise RuntimeError("Training cancelled by user.")

        print("\n🚀 Starting training...\n")

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
        losses = []
        mtp_losses = []

        for k, pred_hidden in enumerate(outputs["mtp_predictions"]):
            shift = k + 1

            if pred_hidden.shape[1] > 0:
                # Pred_hidden corresponds to positions [0, seq_len - shift - 1]
                # Target at shift + i is predicted by pred_hidden at i
                target_shifted = batch["target_hidden"][:, shift:shift + pred_hidden.shape[1]]
                mask_shifted = batch["loss_mask"][:, shift:shift + pred_hidden.shape[1]]

                # Ensure shapes match
                min_len = min(pred_hidden.shape[1], target_shifted.shape[1])

                if min_len > 0:
                    pred_trimmed = pred_hidden[:, :min_len]
                    target_trimmed = target_shifted[:, :min_len]
                    mask_trimmed = mask_shifted[:, :min_len]

                    loss_k = masked_mse_loss(pred_trimmed, target_trimmed, mask_trimmed)
                    print(f"DEBUG: k={k}, pred shape={pred_hidden.shape}, target shape={target_shifted.shape}, loss_k={loss_k}, requires_grad={loss_k.requires_grad}")
                    losses.append(loss_k)
                    mtp_losses.append(loss_k.item())

        if losses:
            total_loss = sum(losses) / len(losses)
            print(f"DEBUG: losses={len(losses)}, total_loss={total_loss}, requires_grad={total_loss.requires_grad}, grad_fn={total_loss.grad_fn}")
        else:
            # No valid losses - return zero loss tensor that requires grad
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            print(f"DEBUG: No losses, using zero tensor")

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
    parser.add_argument("--skip-hardware-check", action="store_true",
                        help="Skip GPU/disk requirements check")

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
        warmup_steps=args.warmup_steps,
        skip_hardware_check=args.skip_hardware_check
    )

    trainer.train()


if __name__ == "__main__":
    main()
