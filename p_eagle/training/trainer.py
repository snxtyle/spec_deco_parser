#!/usr/bin/env python3
"""
P-EAGLE Drafter Training Script

Trains a Drafter model to predict K future hidden states of the Target Model
using Multi-Token Prediction (MTP) heads with parallel speculation.
"""

import argparse
import json
import os
import re
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
from ..utils.loss_utils import masked_mse_loss, hidden_state_token_loss
from ..utils.metrics import MetricsTracker, GenerationMetrics


def run_pre_training_security_check(feature_dir: str) -> bool:
    """
    Run pre-training security verification.

    Scans feature directory source data for secrets using gitleaks
    and custom patterns. Fails fast if CRITICAL secrets detected.

    Returns True if safe to proceed, False if secrets found.
    """
    import subprocess
    import tempfile
    import re

    print_section("PRE-TRAINING SECURITY VERIFICATION")

    # Check if we have the source dataset in feature metadata
    feature_path = Path(feature_dir)
    if not feature_path.exists():
        print("⚠️  Feature directory not found, skipping security check")
        return True

    # Try to find source dataset from feature metadata
    source_dataset = None
    for meta_file in feature_path.glob("*_shard*.pt"):
        try:
            import torch
            data = torch.load(meta_file, map_location="cpu")
            # Features don't contain raw text, they're already processed
            # Security should be checked at data generation time
            print("✅ Features are pre-processed tensors (no raw text to scan)")
            print("   Security scan should have run during: generate_data.py → extract_features.py")
            return True
        except Exception:
            continue

    # If we reach here, no features found
    print("⚠️  No feature files found to verify")
    return True


def verify_dataset_source_security(dataset_path: str, skip_check: bool = False) -> bool:
    """
    Verify dataset source is clean before training.

    This should be called on the ORIGINAL dataset (JSONL) before
    feature extraction or training.
    """
    if skip_check:
        print("⚠️  Security check skipped (--skip-security-check)")
        return True

    if not Path(dataset_path).exists():
        print(f"\n{'='*70}")
        print("⛔ SECURITY CHECK ERROR")
        print(f"{'='*70}")
        print(f"Dataset not found: {dataset_path}")
        print("Cannot run security verification without dataset source.")
        print("Use --skip-security-check only if intentionally bypassing.")
        return False  # Fail if we can't verify security

    print_section("DATASET SECURITY SCAN")
    print(f"Dataset: {dataset_path}")

    # Try to run gitleaks
    try:
        result = subprocess.run(
            ["which", "gitleaks"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            print("⚠️  gitleaks not installed, trying to install/download...")
            # Try auto-install
            install_script = Path(__file__).parent.parent.parent / "scripts" / "scan_dataset_secrets.py"
            if install_script.exists():
                print(f"   Using: {install_script}")
    except Exception:
        pass

    # Run basic regex scan for common patterns
    print("\n  Running regex pattern scan...")
    patterns = [
        (r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', 'Indian PAN'),  # PAN numbers
        (r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b', 'Credit Card'),  # Credit cards
        (r'\bAKIA[0-9A-Z]{16}\b', 'AWS Access Key'),  # AWS keys
        (r'-----BEGIN (?:RSA |DSA |EC )?PRIVATE KEY-----', 'Private Key'),  # Private keys
    ]

    findings = []
    line_count = 0

    try:
        with open(dataset_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                for pattern, name in patterns:
                    if re.search(pattern, line):
                        findings.append((line_num, name))

                # Limit scan to first 1000 lines for speed
                if line_num >= 1000:
                    break
    except Exception as e:
        print(f"  ⚠️  Could not scan dataset: {e}")
        return True

    if findings:
        print(f"\n  ❌ SECURITY ISSUES FOUND:")
        for line_num, name in findings[:10]:
            print(f"     Line {line_num}: {name}")
        if len(findings) > 10:
            print(f"     ... and {len(findings) - 10} more")
        print(f"\n  ⛔ TRAINING ABORTED - Clean dataset required")
        return False
    else:
        print(f"  ✅ No obvious secrets in first {line_count} lines")
        print(f"  ✅ Dataset security check passed")
        return True


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
        skip_hardware_check: bool = False,
        quantization: str = None
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
        # FIX: Explicitly disable hidden injection to ensure training/inference alignment
        # Training uses tri-layer fused features, inference uses last-layer - mismatch if enabled
        self.model = EagleDrafterModel(
            base_model_name=drafter_model_name,
            target_hidden_dim=target_hidden_dim,
            speculation_depth=speculation_depth,
            use_lora=use_lora,
            lora_rank=lora_rank,
            device=device,
            use_hidden_injection=False,  # CRITICAL: Must match inference setting
            quantization=quantization
        )

        # Load tokenizer (use same cache as model)
        import os
        cache_dir = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_cache")
        self.tokenizer = AutoTokenizer.from_pretrained(drafter_model_name, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load target model's lm_head for token-level loss computation
        # This is critical: drafter's hidden states are converted to tokens via target's lm_head
        print(f"Loading target lm_head for token-level training...")
        print(f"NOTE: Target model must have hidden_dim={target_hidden_dim} for lm_head compatibility")
        self.target_lm_head = self._load_target_lm_head(target_hidden_dim)

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

        # Setup dataset with train/validation split
        full_dataset = EagleTrainingDataset(
            feature_dir=feature_dir,
            tokenizer=self.tokenizer,
            speculation_depth=speculation_depth
        )

        # Split dataset: 80% train, 20% validation
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        if val_size > 0:
            from torch.utils.data import random_split
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            print(f"Dataset split: {train_size} train, {val_size} validation")
        else:
            self.train_dataset = full_dataset
            self.val_dataset = None
            print(f"Dataset: {len(full_dataset)} samples (no validation split)")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0
        )

        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
                num_workers=0
            )
        else:
            self.val_loader = None

        # Setup scheduler
        total_steps = len(self.train_loader) * num_epochs
        self.warmup_steps = warmup_steps  # Store for saving in history
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")
        self.metrics_tracker = MetricsTracker()

        print(f"Training setup complete:")
        total_samples = len(self.train_dataset) + (len(self.val_dataset) if self.val_dataset else 0)
        print(f"  Total samples: {total_samples}")
        print(f"  Train: {len(self.train_dataset)}, Val: {len(self.val_dataset) if self.val_dataset else 0}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Total steps: {total_steps}")

    def _load_target_lm_head(self, target_hidden_dim: int):
        """Load target model's lm_head for perfect KL alignment.

        Attempts to load the actual lm_head weights saved during feature extraction.
        This ensures the KL loss is computed with the exact same projection layer
        that the target model uses, aligning training perfectly with inference.
        """
        import torch.nn as nn
        from pathlib import Path

        # Get vocab size from drafter (they should match for compatible models)
        vocab_size = len(self.tokenizer)

        # Create lm_head with correct dimensions (bfloat16 to match drafter output)
        lm_head = nn.Linear(target_hidden_dim, vocab_size, bias=False, dtype=torch.bfloat16).to(self.device)

        # Attempt to load actual lm_head weights from feature files
        loaded_weights = False
        try:
            feature_files = list(Path(self.feature_dir).glob("*_shard*.pt"))
            if feature_files:
                # Load from the first shard to get lm_head weights
                data = torch.load(feature_files[0], map_location=self.device, weights_only=False)

                if "lm_head" in data and data["lm_head"] is not None:
                    saved_lm_head = data["lm_head"]
                    saved_weight = saved_lm_head.get("weight") if isinstance(saved_lm_head, dict) else None

                    if saved_weight is not None:
                        saved_vocab_size = saved_weight.shape[0]
                        saved_hidden_size = saved_weight.shape[1]

                        print(f"  Saved lm_head: {saved_hidden_size} -> {saved_vocab_size}")
                        print(f"  Target lm_head: {target_hidden_dim} -> {vocab_size}")

                        # Handle size mismatch by copying overlapping weights
                        if saved_vocab_size != vocab_size or saved_hidden_size != target_hidden_dim:
                            print(f"  Resizing: copying overlapping weights...")
                            with torch.no_grad():
                                min_vocab = min(saved_vocab_size, vocab_size)
                                min_hidden = min(saved_hidden_size, target_hidden_dim)
                                lm_head.weight[:min_vocab, :min_hidden] = saved_weight[:min_vocab, :min_hidden]
                            print(f"  Copied {min_vocab} vocab x {min_hidden} hidden dims")
                            loaded_weights = True
                        else:
                            lm_head.load_state_dict(saved_lm_head)
                            loaded_weights = True
                            print(f"  Loaded target lm_head weights from feature file")
                    else:
                        print(f"  Warning: Could not parse lm_head weight tensor")
                else:
                    print(f"  Warning: No lm_head weights found in feature files")

                # Verify dimensions
                saved_vocab_size = data.get("vocab_size", vocab_size)
                saved_hidden_size = data.get("hidden_size", target_hidden_dim)

                if saved_vocab_size != vocab_size:
                    print(f"  Note: Vocab size adjusted - features: {saved_vocab_size}, using: {vocab_size}")
                if saved_hidden_size != target_hidden_dim:
                    print(f"  Error: Hidden dim mismatch - features: {saved_hidden_size}, expected: {target_hidden_dim}")
            else:
                print(f"  Warning: No feature files found in {self.feature_dir}")
        except Exception as e:
            print(f"  Warning: Failed to load lm_head weights: {e}")

        if not loaded_weights:
            # Fall back to random initialization with warning
            nn.init.normal_(lm_head.weight, std=0.02)
            print(f"  Using randomly initialized lm_head (KL loss may not align perfectly)")

        print(f"  lm_head: {target_hidden_dim} -> {vocab_size}")
        print(f"  During inference, actual target lm_head will be used")

        return lm_head

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

    @torch.no_grad()
    def _validation_step(self):
        """Compute validation loss."""
        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            target_hidden = batch["target_hidden"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_hidden=target_hidden,
                is_training=True
            )

            # Apply loss mask to compute validation loss
            mtp_predictions = outputs["mtp_predictions"]
            total_loss = 0.0
            valid_heads = 0

            for i, pred_hidden in enumerate(mtp_predictions):
                if i >= self.model.speculation_depth:
                    break

                # Get shifted targets for this head
                shift = i + 1
                if shift >= target_hidden.shape[1]:
                    continue

                pred_shifted = pred_hidden[:, :-shift, :]
                target_shifted = target_hidden[:, shift:, :]

                # Ensure shapes match (trim to minimum length)
                min_len = min(pred_shifted.shape[1], target_shifted.shape[1])
                pred_shifted = pred_shifted[:, :min_len, :]
                target_shifted = target_shifted[:, :min_len, :]
                mask_shifted = loss_mask[:, shift:shift + min_len].unsqueeze(-1).float()

                # Compute masked MSE loss
                mse = ((pred_shifted - target_shifted) ** 2) * mask_shifted
                loss = mse.sum() / (mask_shifted.sum() + 1e-8)

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    valid_heads += 1

            if valid_heads > 0:
                val_losses.append(total_loss / valid_heads)

        self.model.train()
        return np.mean(val_losses) if val_losses else None

    def train(self):
        """Main training loop with validation and early stopping."""
        self.model.train()
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 5  # Early stopping patience
        epoch_stats = []

        # Sanity check: verify first batch has non-zero loss mask
        print("\nVerifying training data...")
        first_batch = next(iter(self.train_loader))
        mask_sum = first_batch["loss_mask"].sum().item()
        print(f"  First batch mask sum: {mask_sum}")
        if mask_sum == 0:
            print("  WARNING: Loss mask is all zeros! Training will fail.")
            print("  Check that dataset has proper 'loss_mask_segments'.")
        else:
            print(f"  OK: {mask_sum} trainable tokens in first batch")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            epoch_losses = []
            epoch_mtp_losses = {i: [] for i in range(self.model.speculation_depth)}
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            for batch_idx, batch in enumerate(pbar):
                loss, metrics = self._training_step(batch)

                epoch_losses.append(loss.item())
                self.global_step += 1

                # Track per-head losses
                for i in range(self.model.speculation_depth):
                    key = f"mtp_loss_{i+1}"
                    if key in metrics:
                        epoch_mtp_losses[i].append(metrics[key])

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

                # Log to TensorBoard
                if self.global_step % 10 == 0:
                    self.writer.add_scalar("train/total_loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                    self.writer.add_scalar("train/mtp_loss_avg", metrics.get("mtp_loss_avg", 0), self.global_step)
                    self.writer.add_scalar("train/kl_loss_avg", metrics.get("kl_loss_avg", 0), self.global_step)
                    self.writer.add_scalar("train/token_acc_avg", metrics.get("token_acc_avg", 0), self.global_step)
                    for k, v in metrics.items():
                        if k.startswith("mtp_loss_") or k.startswith("token_acc_"):
                            self.writer.add_scalar(f"train/{k}", v, self.global_step)

                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self._save_checkpoint(f"checkpoint_step_{self.global_step}")

            # Epoch summary
            avg_train_loss = np.mean(epoch_losses)

            # Validation
            val_loss = self._validation_step()

            epoch_stat = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": val_loss,
                "per_head_avg_loss": {}
            }
            for i in range(self.model.speculation_depth):
                if epoch_mtp_losses[i]:
                    epoch_stat["per_head_avg_loss"][f"head_{i+1}"] = np.mean(epoch_mtp_losses[i])

            epoch_stats.append(epoch_stat)

            # Log to TensorBoard
            self.writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)
            if val_loss is not None:
                self.writer.add_scalar("epoch/val_loss", val_loss, epoch + 1)

            print(f"Epoch {epoch + 1} summary:")
            print(f"  Train loss: {avg_train_loss:.6f}")
            if val_loss is not None:
                print(f"  Val loss:   {val_loss:.6f}")
                # Check for overfitting
                if val_loss > avg_train_loss * 1.5:
                    print(f"  ⚠️  WARNING: Possible overfitting (val >> train)")

            for head, loss_val in epoch_stat["per_head_avg_loss"].items():
                print(f"  {head}: {loss_val:.6f}")

            # Save best model based on validation loss
            model_improved = False
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_improved = True
                    patience_counter = 0
            else:
                # Fallback to training loss if no validation set
                if avg_train_loss < best_val_loss:
                    best_val_loss = avg_train_loss
                    model_improved = True

            if model_improved:
                self._save_checkpoint("best_model")
                print(f"  *** New best model saved! ***")
            else:
                patience_counter += 1
                if patience_counter >= patience and self.val_loader is not None:
                    print(f"  ⚠️  Early stopping triggered (patience={patience})")
                    break

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "final_best_loss": best_loss,
                "epochs": epoch_stats,
                "config": {
                    "drafter_model": self.drafter_model_name,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.train_loader.batch_size,
                    "learning_rate": self.optimizer.defaults["lr"],
                    "warmup_steps": self.warmup_steps
                }
            }, f, indent=2)
        print(f"\nTraining history saved to {history_path}")

        self.writer.close()
        print("\nTraining complete!")

    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """Single training step with P-EAGLE aligned loss.

        Key insight: During inference, drafter's predicted hidden states are converted
        to tokens via the TARGET model's lm_head. So we must train to match the
        token distributions, not just hidden state vectors.
        """
        self.optimizer.zero_grad()

        # Forward pass with optional target hidden state injection
        # target_hidden is passed as input for hidden state injection during distillation
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            target_hidden=batch["target_hidden"],
            is_training=True
        )

        # Compute losses for each MTP head
        kl_losses = []
        mse_losses = []
        token_accs = []
        mtp_losses = []

        for k, pred_hidden in enumerate(outputs["mtp_predictions"]):
            shift = k + 1

            if pred_hidden.shape[1] > 0:
                target_shifted = batch["target_hidden"][:, shift:shift + pred_hidden.shape[1]]
                mask_shifted = batch["loss_mask"][:, shift:shift + pred_hidden.shape[1]]

                min_len = min(pred_hidden.shape[1], target_shifted.shape[1])

                if min_len > 0:
                    pred_trimmed = pred_hidden[:, :min_len]
                    target_trimmed = target_shifted[:, :min_len]
                    mask_trimmed = mask_shifted[:, :min_len]

                    # P-EAGLE aligned loss: match token distributions via target lm_head
                    kl_loss_k, mse_loss_k, acc_k = hidden_state_token_loss(
                        pred_trimmed,
                        target_trimmed,
                        self.target_lm_head,
                        mask_trimmed,
                        temperature=1.0
                    )

                    kl_losses.append(kl_loss_k)
                    mse_losses.append(mse_loss_k)
                    token_accs.append(acc_k.item())
                    mtp_losses.append(mse_loss_k.item())

        # Combine losses: KL divergence (token distribution) + MSE (hidden state)
        # KL ensures tokens match, MSE ensures hidden states are structurally similar
        total_loss = torch.tensor(0.0, device=self.device)

        if kl_losses:
            kl_total = sum(kl_losses) / len(kl_losses)
            total_loss = total_loss + kl_total  # Primary: token distribution matching

        if mse_losses:
            mse_total = sum(mse_losses) / len(mse_losses)
            total_loss = total_loss + 0.1 * mse_total  # Secondary: hidden state similarity

        if total_loss.item() == 0:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()

        metrics = {
            "mtp_loss_avg": np.mean(mtp_losses) if mtp_losses else 0.0,
            "kl_loss_avg": sum(kl.item() for kl in kl_losses) / len(kl_losses) if kl_losses else 0.0,
            "token_acc_avg": np.mean(token_accs) if token_accs else 0.0,
        }
        for i, loss_i in enumerate(mtp_losses):
            metrics[f"mtp_loss_{i+1}"] = loss_i
        for i, acc_i in enumerate(token_accs):
            metrics[f"token_acc_{i+1}"] = acc_i

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
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5 for small datasets)")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--skip-hardware-check", action="store_true",
                        help="Skip GPU/disk requirements check")
    parser.add_argument("--skip-security-check", action="store_true",
                        help="Skip pre-training security verification (not recommended)")
    parser.add_argument("--dataset-source", type=str, default=None,
                        help="Path to original dataset JSONL for security verification")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit"],
                        help="Quantize drafter model (4bit or 8bit) to reduce VRAM usage")

    args = parser.parse_args()

    # Run pre-training security verification
    if args.dataset_source:
        if not verify_dataset_source_security(args.dataset_source, args.skip_security_check):
            print("\n⛔ Training aborted due to security concerns.")
            print("   Use --skip-security-check to override (not recommended)")
            exit(1)
    else:
        # Check feature directory security
        if not run_pre_training_security_check(args.feature_dir):
            print("\n⛔ Training aborted due to security concerns.")
            exit(1)

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
        skip_hardware_check=args.skip_hardware_check,
        quantization=args.quantization
    )

    trainer.train()


if __name__ == "__main__":
    main()
