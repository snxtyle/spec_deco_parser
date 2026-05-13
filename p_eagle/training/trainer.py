#!/usr/bin/env python3
"""
P-EAGLE Drafter Training Script

Trains a Drafter model to predict K future hidden states of the Target Model
using Multi-Token Prediction (MTP) heads with parallel speculation.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import timedelta, datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
from bitsandbytes.optim import PagedAdamW8bit
from tqdm import tqdm
import numpy as np

from ..models.peagle_drafter import EagleDrafterModel
from ..utils.feature_utils import EagleTrainingDataset
from ..utils.loss_utils import masked_mse_loss, hidden_state_token_loss
from ..utils.metrics import MetricsTracker, GenerationMetrics


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )

        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_training_logger(output_dir: Path, run_name: str = None) -> logging.Logger:
    """Setup comprehensive logging for training runs.

    Creates timestamped log files and captures both console and file output.

    Args:
        output_dir: Directory to save logs
        run_name: Optional run name, defaults to timestamp

    Returns:
        Logger instance configured for training
    """
    # Create logs directory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped run identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"training_{timestamp}"

    # Create run-specific log directory
    run_log_dir = logs_dir / run_id
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = logging.getLogger("peagle_training")
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler - main log
    log_file = run_log_dir / "training.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Also capture raw stdout/stderr to a separate file
    raw_log_file = run_log_dir / "output.log"
    raw_fh = logging.FileHandler(raw_log_file, mode='w')
    raw_fh.setLevel(logging.DEBUG)
    raw_fh.setFormatter(logging.Formatter("%(message)s"))
    raw_fh.addFilter(lambda record: True)  # Capture everything
    logger.addHandler(raw_fh)

    logger.info(f"=" * 70)
    logger.info(f"P-EAGLE TRAINING SESSION STARTED")
    logger.info(f"=" * 70)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Log directory: {run_log_dir}")
    logger.info(f"Main log file: {log_file}")
    logger.info(f"Raw output file: {raw_log_file}")
    logger.info(f"=" * 70)

    return logger, run_log_dir, run_id


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


class GPUMemoryMonitor:
    """Monitors GPU memory usage and prevents OOM crashes.

    Tracks memory allocation in real-time and can trigger emergency
    measures (clearing cache, reducing batch size, or stopping training)
    when memory limits are approached.
    """

    def __init__(self, device: str = "cuda", safety_margin_gb: float = 1.0, logger: logging.Logger = None):
        self.device = device
        self.safety_margin_gb = safety_margin_gb
        self.logger = logger or logging.getLogger("peagle_training")
        self.max_allocated_gb = 0.0
        self.oom_count = 0
        self.emergency_reduced_batch = False

        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available - GPU monitoring disabled")
            self.enabled = False
        else:
            self.enabled = True
            self.device_count = torch.cuda.device_count()
            self._log_gpu_info()

    def _log_gpu_info(self):
        """Log GPU information at startup."""
        for i in range(self.device_count):
            props = torch.cuda.get_device_properties(i)
            total_gb = props.total_memory / (1024**3)
            self.logger.info(f"GPU {i}: {props.name} | Total Memory: {total_gb:.2f} GB")

    def get_memory_stats(self, device_index: int = 0) -> Dict[str, float]:
        """Get current memory statistics for a GPU."""
        if not self.enabled:
            return {}

        torch.cuda.synchronize(device_index)

        allocated = torch.cuda.memory_allocated(device_index) / (1024**3)
        reserved = torch.cuda.memory_reserved(device_index) / (1024**3)
        total = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
        free = total - allocated

        # Track peak usage
        self.max_allocated_gb = max(self.max_allocated_gb, allocated)

        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "total_gb": total,
            "free_gb": free,
            "utilization_percent": (allocated / total) * 100
        }

    def check_memory(self, device_index: int = 0) -> Tuple[bool, str]:
        """Check if GPU memory is within safe limits.

        Returns:
            (is_safe, message) tuple
        """
        if not self.enabled:
            return True, "GPU monitoring disabled"

        stats = self.get_memory_stats(device_index)

        if stats["free_gb"] < self.safety_margin_gb:
            return False, f"Low memory: {stats['free_gb']:.2f} GB free (safety margin: {self.safety_margin_gb} GB)"

        if stats["utilization_percent"] > 95:
            return False, f"High utilization: {stats['utilization_percent']:.1f}%"

        return True, f"OK: {stats['allocated_gb']:.2f} GB allocated, {stats['free_gb']:.2f} GB free"

    def emergency_cleanup(self) -> bool:
        """Perform emergency memory cleanup.

        Returns:
            True if cleanup was successful and training can continue
        """
        if not self.enabled:
            return True

        self.logger.warning("🚨 EMERGENCY GPU MEMORY CLEANUP INITIATED")

        # Empty CUDA cache
        torch.cuda.empty_cache()
        self.logger.info("  - Emptied CUDA cache")

        # Force garbage collection
        import gc
        gc.collect()
        self.logger.info("  - Ran garbage collection")

        # Check memory after cleanup
        stats = self.get_memory_stats()
        self.logger.info(f"  - Free memory after cleanup: {stats['free_gb']:.2f} GB")

        if stats["free_gb"] < self.safety_margin_gb:
            self.logger.error("  - Cleanup insufficient - still below safety margin")
            return False

        self.oom_count += 1
        self.logger.warning(f"  - Emergency cleanup #{self.oom_count} successful")
        return True

    def log_memory_summary(self):
        """Log a summary of memory usage."""
        if not self.enabled:
            return

        stats = self.get_memory_stats()
        self.logger.info(
            f"GPU Memory: {stats['allocated_gb']:.2f} GB allocated | "
            f"{stats['free_gb']:.2f} GB free | "
            f"Peak: {self.max_allocated_gb:.2f} GB"
        )

    def get_memory_report(self) -> Dict[str, Any]:
        """Get a comprehensive memory report for saving to logs."""
        if not self.enabled:
            return {"enabled": False}

        stats = self.get_memory_stats()
        return {
            "enabled": True,
            "peak_allocated_gb": self.max_allocated_gb,
            "oom_incidents": self.oom_count,
            "emergency_batch_reduction": self.emergency_reduced_batch,
            "current_stats": stats
        }


def oom_recovery_handler(func):
    """Decorator to catch OOM errors and attempt recovery.

    Wraps training functions to catch CUDA OOM errors, attempt cleanup,
    and potentially retry with reduced memory usage.
    """
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                self.logger.error(f"🚨 CUDA OOM ERROR: {e}")

                # Try emergency cleanup
                if hasattr(self, 'gpu_monitor') and self.gpu_monitor.emergency_cleanup():
                    self.logger.warning("Attempting to continue after OOM cleanup...")
                    # Re-raise to let caller decide whether to retry
                    raise RuntimeError(f"OOM recovered but step failed: {e}")
                else:
                    self.logger.error("OOM cleanup failed - stopping training")
                    # Log final memory state
                    if hasattr(self, 'gpu_monitor'):
                        self.gpu_monitor.log_memory_summary()
                    raise RuntimeError(f"Fatal OOM: {e}")
            else:
                raise
    return wrapper


def print_section(title: str, logger: logging.Logger = None):
    """Print formatted section header."""
    lines = [
        "",
        f"{'='*70}",
        f"  {title}",
        f"{'='*70}"
    ]
    if logger:
        for line in lines:
            logger.info(line)
    else:
        for line in lines:
            print(line)


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
        yes: bool = False,
        quantization: str = None,
        logger: logging.Logger = None,
        run_log_dir: Path = None,
        gpu_safety_margin_gb: float = 1.5,
        resume_from: str = None,
        gradient_accumulation_steps: int = 1,
        max_seq_len: int = 2048,
        rank: int = 0,
        world_size: int = 1
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir = feature_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.save_every = save_every
        self.device = device
        self.yes = yes
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        self.drafter_model_name = drafter_model_name
        self.logger = logger or logging.getLogger("peagle_training")
        self.run_log_dir = run_log_dir
        self.resume_from = resume_from
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = (rank == 0)

        # Initialize GPU memory monitor
        self.gpu_monitor = GPUMemoryMonitor(
            device=device,
            safety_margin_gb=gpu_safety_margin_gb,
            logger=self.logger
        )

        # Hardware requirement check
        if not skip_hardware_check:
            self._run_hardware_check(
                drafter_model_name=drafter_model_name,
                target_hidden_dim=target_hidden_dim,
                batch_size=batch_size,
                speculation_depth=speculation_depth,
                use_lora=use_lora,
                num_epochs=num_epochs,
                auto_confirm=self.yes
            )

        # Initialize model
        # EAGLE-3 requires hidden injection via CONCATENATION at first layer
        # First layer accepts 2x hidden size: [embeds; target_hidden]
        self.model = EagleDrafterModel(
            base_model_name=drafter_model_name,
            target_hidden_dim=target_hidden_dim,
            speculation_depth=speculation_depth,
            use_lora=use_lora,
            lora_rank=lora_rank,
            device=device,
            use_hidden_injection=True,   # CRITICAL: EAGLE-3 needs target hidden state injection
            injection_mode="concat",     # CONCATENATION: first layer gets 2x hidden size input
            quantization=quantization
        )

        # Wrap model with DDP if using multiple GPUs
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)
            if self.is_main_process:
                self.logger.info(f"Wrapped model with DDP (world_size={self.world_size})")

        # Load checkpoint if resuming (only on main process)
        if self.resume_from and self.is_main_process:
            self.logger.info(f"Resuming training from checkpoint: {self.resume_from}")
            # Unwrap DDP if needed to load checkpoint
            model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
            model_to_load.load_checkpoint(self.resume_from, device=device)
            # Extract step number from checkpoint name (e.g. checkpoint_step_1000)
            import re
            step_match = re.search(r'step_(\d+)', self.resume_from)
            if step_match:
                self.global_step = int(step_match.group(1))
                if self.is_main_process:
                    self.logger.info(f"Resumed from step {self.global_step}")
            else:
                if self.is_main_process:
                    self.logger.warning("Could not extract step number from checkpoint name")

        # === SPEED OPTIMIZATIONS ===
        # Enable TF32 for faster matmul (10% speedup, no quality loss)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  Enabled TF32 for faster GPU computation")

        # NOTE: torch.compile disabled for EAGLE-3 compatibility
        # Gemma3's rotary embeddings have complex internal state that
        # doesn't work well with torch.compile's dynamo tracing
        print("  Skipping torch.compile (disabled for EAGLE-3 compatibility)")
        # self.model = torch.compile(self.model, mode="reduce-overhead")
        # ==========================

        # Enable gradient checkpointing to save VRAM (trades compute for memory)
        if hasattr(self.model.base_model, 'gradient_checkpointing_enable'):
            self.model.base_model.gradient_checkpointing_enable()
            print("  Enabled gradient checkpointing (saves ~50% VRAM)")

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

        # Setup optimizer with SEPARATE learning rates for different components
        # CRITICAL FIX: MTP heads need higher LR to learn effectively
        print("Setting up PagedAdamW8bit optimizer with separate LR groups...")

        lora_params = []
        mtp_params = []
        proj_params = []
        bias_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "bias" in name or "norm" in name:
                bias_params.append(param)
            elif "mtp_heads" in name:
                mtp_params.append(param)
            elif "dim_projection" in name:
                proj_params.append(param)
            elif "lora" in name.lower():
                lora_params.append(param)
            else:
                # Other trainable params
                bias_params.append(param)

        # Higher LR for MTP heads (10x) to ensure they learn
        # LoRA uses base LR
        print(f"  Parameter groups:")
        print(f"    LoRA params: {len(lora_params)} tensors")
        print(f"    MTP heads: {len(mtp_params)} tensors (LR = {learning_rate * 10:.2e})")
        print(f"    Projection: {len(proj_params)} tensors (LR = {learning_rate * 10:.2e})")
        print(f"    Bias/No-WD: {len(bias_params)} tensors")

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": learning_rate, "weight_decay": 0.01})
        if mtp_params:
            param_groups.append({"params": mtp_params, "lr": learning_rate * 10, "weight_decay": 0.01})
        if proj_params:
            param_groups.append({"params": proj_params, "lr": learning_rate * 10, "weight_decay": 0.01})
        if bias_params:
            param_groups.append({"params": bias_params, "lr": learning_rate, "weight_decay": 0.0})

        self.optimizer = PagedAdamW8bit(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Load dataset with lazy shard loading (official EAGLE pattern)
        # Dataset loads only one shard at a time, reducing RAM from 148GB to ~6GB
        from ..utils.feature_utils import EagleTrainingDataset

        self.train_dataset = EagleTrainingDataset(
            feature_dir=feature_dir,
            tokenizer=self.tokenizer,
            speculation_depth=speculation_depth,
            max_seq_len=max_seq_len,
            shard_cache_size=1  # Keep only 1 shard in RAM at a time
        )

        # Split into train/val (95/5 split like official EAGLE)
        total_samples = len(self.train_dataset)
        val_size = int(0.05 * total_samples)
        train_size = total_samples - val_size

        from torch.utils.data import random_split
        generator = torch.Generator().manual_seed(42)
        self.train_subset, self.val_subset = random_split(
            self.train_dataset, [train_size, val_size], generator=generator
        )

        print(f"Dataset split: {train_size} train, {val_size} validation")

        # Create samplers for distributed training
        if self.world_size > 1:
            self.train_sampler = DistributedSampler(
                self.train_subset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            self.val_sampler = DistributedSampler(
                self.val_subset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False
            )
        else:
            self.train_sampler = None
            self.val_sampler = None

        # Create dataloaders
        # Note: pin_memory=False because we move tensors to device in training loop
        self.train_loader = DataLoader(
            self.train_subset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),  # Shuffle only if no sampler
            sampler=self.train_sampler,
            collate_fn=self._collate_fn,
            num_workers=0,  # Must be 0 for lazy loading to work correctly
            pin_memory=False
        )

        self.val_loader = DataLoader(
            self.val_subset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.val_sampler,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=False
        )

        self.speculation_depth = speculation_depth

        # Setup scheduler
        total_steps = len(self.train_loader) * num_epochs
        self.warmup_steps = warmup_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.output_dir / "logs")
        self.metrics_tracker = MetricsTracker()

        print(f"Training setup complete:")
        print(f"  Total samples: {total_samples}")
        print(f"  Train: {train_size}, Val: {val_size}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Steps per epoch: {len(self.train_loader)}")

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
        num_epochs: int,
        auto_confirm: bool = False
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

        if auto_confirm:
            print("\n🚀 Auto-confirmed ( --yes flag ), starting training...\n")
        else:
            user_input = input("\nStart training? [Y/n]: ").strip().lower()
            if user_input == 'n':
                raise RuntimeError("Training cancelled by user.")

        print("\n🚀 Starting training...\n")

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for batching. Returns CPU tensors."""
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

        # Return CPU tensors - moved to device in training loop
        return {
            "input_ids": input_ids,
            "target_hidden": target_hidden,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask
        }

    @torch.no_grad()
    def _validation_step(self, epoch: int = 1):
        """Compute validation loss using same loss function as training.

        Uses curriculum learning to only validate active heads.
        """
        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = []

        # Determine active heads for curriculum learning
        active_heads = self._get_active_heads(epoch)

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
            ce_losses = []
            mse_losses = []

            for i, pred_hidden in enumerate(mtp_predictions):
                # CURRICULUM: Only validate active heads
                if i >= active_heads:
                    break

                # Get shifted targets for this head
                shift = i + 1
                if shift >= target_hidden.shape[1]:
                    continue

                pred_shifted = pred_hidden[:, :-shift, :]
                target_shifted = target_hidden[:, shift:, :]

                # Ensure shapes match (trim to minimum length)
                min_len = min(pred_shifted.shape[1], target_shifted.shape[1])
                if min_len <= 0:
                    continue

                pred_trimmed = pred_shifted[:, :min_len, :]
                target_trimmed = target_shifted[:, :min_len, :]
                mask_trimmed = loss_mask[:, shift:shift + min_len]

                # Use SAME loss function as training for consistency
                ce_loss_k, mse_loss_k, _ = hidden_state_token_loss(
                    pred_trimmed,
                    target_trimmed,
                    self.target_lm_head,
                    mask_trimmed,
                    temperature=1.0
                )

                if not torch.isnan(ce_loss_k):
                    ce_losses.append(ce_loss_k.item())
                    mse_losses.append(mse_loss_k.item())

            # Combine losses same way as training
            total_loss = 0.0
            if ce_losses:
                # Apply same weighting as training
                weighted_ce = []
                for i, ce in enumerate(ce_losses):
                    weight = max(0.5, 1.0 - i * 0.1)
                    weighted_ce.append(ce * weight)
                ce_total = sum(weighted_ce) / sum(max(0.5, 1.0 - i * 0.1) for i in range(len(weighted_ce)))
                total_loss += ce_total

            if mse_losses:
                mse_total = sum(mse_losses) / len(mse_losses)
                total_loss += 0.1 * mse_total

            # Apply same scaling as training
            total_loss *= 10.0

            val_losses.append(total_loss)

        self.model.train()
        return np.mean(val_losses) if val_losses else None

    def train(self):
        """Main training loop with validation and early stopping."""
        self.model.train()
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 5  # Early stopping patience
        epoch_stats = []

        # Initial GPU memory check
        self.logger.info(f"\n{'='*50}")
        self.logger.info("INITIAL GPU MEMORY CHECK")
        self.logger.info(f"{'='*50}")
        is_safe, mem_msg = self.gpu_monitor.check_memory()
        if not is_safe:
            self.logger.warning(f"Initial memory warning: {mem_msg}")
            if not self.gpu_monitor.emergency_cleanup():
                self.logger.error("Insufficient GPU memory to start training")
                raise RuntimeError("GPU memory too low to begin training")
        self.gpu_monitor.log_memory_summary()

        # Sanity check: verify first batch has non-zero loss mask
        self.logger.info("\nVerifying training data...")
        first_batch = next(iter(self.train_loader))
        mask_sum = first_batch["loss_mask"].sum().item()
        self.logger.info(f"  First batch mask sum: {mask_sum}")
        if mask_sum == 0:
            self.logger.warning("  WARNING: Loss mask is all zeros! Training will fail.")
            self.logger.warning("  Check that dataset has proper 'loss_mask_segments'.")
        else:
            self.logger.info(f"  OK: {mask_sum} trainable tokens in first batch")

        for epoch in range(self.num_epochs):
            # Set epoch for distributed sampler to ensure different shuffling each epoch
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.logger.info(f"{'='*50}")

            # Log curriculum learning status (only on main process)
            active_heads = self._get_active_heads(epoch + 1)
            if self.is_main_process:
                self.logger.info(f"Curriculum: Training heads 1-{active_heads} of {self.model.speculation_depth}")

            epoch_losses = []
            epoch_ce_losses = []
            epoch_mse_losses = []
            epoch_mtp_losses = {i: [] for i in range(self.model.speculation_depth)}
            # Gradient accumulation setup
            grad_accum_steps = self.gradient_accumulation_steps
            effective_batch_size = self.batch_size * grad_accum_steps

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

            # Periodic memory check interval (check 10 times per epoch)
            memory_check_interval = max(1, len(self.train_loader) // 10)
            batchOOM_retry_count = 0

            # Convert iterator to list to support slicing for accumulation
            batch_iterator = iter(self.train_loader)
            batch_idx = 0
            accumulation_counter = 0

            while batch_idx < len(self.train_loader):
                # Check GPU memory periodically
                if batch_idx % memory_check_interval == 0:
                    is_safe, mem_msg = self.gpu_monitor.check_memory()
                    if not is_safe:
                        self.logger.warning(f"Memory warning: {mem_msg}")
                        if not self.gpu_monitor.emergency_cleanup():
                            self.logger.error("Unable to free sufficient memory - stopping training")
                            self._save_checkpoint("emergency_stop_low_memory")
                            return  # Exit training cleanly

                # Accumulate gradients over multiple batches
                accum_loss = 0.0
                accum_metrics = {}
                valid_accum_steps = 0

                for accum_step in range(grad_accum_steps):
                    try:
                        batch = next(batch_iterator)
                    except StopIteration:
                        break

                    current_accum_step = (accumulation_counter * grad_accum_steps + accum_step) % grad_accum_steps
                    is_last_accum = (accum_step == grad_accum_steps - 1)

                    # Perform training step (with OOM recovery)
                    try:
                        loss, metrics = self._training_step(
                            batch,
                            epoch=epoch + 1,
                            accumulation_step=current_accum_step,
                            total_accumulation_steps=grad_accum_steps
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            self.logger.error(f"OOM during training step at batch {batch_idx}, accum_step {accum_step}")
                            if self.gpu_monitor.emergency_cleanup() and batchOOM_retry_count < 3:
                                batchOOM_retry_count += 1
                                self.logger.warning(f"Retrying batch after OOM cleanup (attempt {batchOOM_retry_count}/3)")
                                torch.cuda.synchronize()
                                continue
                            else:
                                self.logger.error("Too many OOM errors - saving checkpoint and stopping")
                                self._save_checkpoint(f"emergency_stop_oom_epoch{epoch+1}_batch{batch_idx}")
                                return
                        else:
                            raise

                    accum_loss += loss.item()
                    valid_accum_steps += 1

                    # Aggregate metrics
                    for key, value in metrics.items():
                        if key not in accum_metrics:
                            accum_metrics[key] = []
                        if hasattr(value, 'item'):
                            accum_metrics[key].append(value.item())
                        else:
                            accum_metrics[key].append(value)

                    batch_idx += 1

                # Average accumulated loss and metrics
                if valid_accum_steps > 0:
                    avg_loss = accum_loss / valid_accum_steps
                    epoch_losses.append(avg_loss)
                    self.global_step += 1
                    accumulation_counter += 1

                # Compute average metrics from accumulation
                avg_metrics = {}
                for key, values in accum_metrics.items():
                    if values:
                        avg_metrics[key] = sum(values) / len(values)

                # Track per-head losses and component losses
                for i in range(self.model.speculation_depth):
                    key = f"mtp_loss_{i+1}"
                    if key in avg_metrics:
                        epoch_mtp_losses[i].append(avg_metrics[key])

                # Track CE and MSE separately for diagnostics
                if "ce_loss_avg" in avg_metrics:
                    epoch_ce_losses.append(avg_metrics["ce_loss_avg"])
                if "mse_loss_avg" in avg_metrics:
                    epoch_mse_losses.append(avg_metrics["mse_loss_avg"])

                # Update progress bar with more informative metrics
                postfix_dict = {
                    "loss": f"{avg_loss:.4f}",
                    "ce": f"{avg_metrics.get('ce_loss_avg', 0):.4f}",
                    "mse": f"{avg_metrics.get('mse_loss_avg', 0):.4f}",
                    "acc": f"{avg_metrics.get('token_acc_avg', 0):.1f}%",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                }
                if grad_accum_steps > 1:
                    postfix_dict["accum"] = f"{valid_accum_steps}/{grad_accum_steps}"
                # Only show mask coverage if it's concerning (low coverage)
                mask_cov = avg_metrics.get('avg_mask_coverage', 1.0)
                if mask_cov < 0.5:
                    postfix_dict["mask"] = f"{mask_cov:.0%}"
                pbar.set_postfix(postfix_dict)
                pbar.update(valid_accum_steps)

                # Log to TensorBoard
                if self.global_step % 10 == 0:
                    self.writer.add_scalar("train/total_loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], self.global_step)
                    self.writer.add_scalar("train/mtp_loss_avg", avg_metrics.get("mtp_loss_avg", 0), self.global_step)
                    self.writer.add_scalar("train/ce_loss_avg", avg_metrics.get("ce_loss_avg", 0), self.global_step)
                    self.writer.add_scalar("train/token_acc_avg", avg_metrics.get("token_acc_avg", 0), self.global_step)
                    for k, v in avg_metrics.items():
                        if k.startswith("mtp_loss_") or k.startswith("token_acc_"):
                            self.writer.add_scalar(f"train/{k}", v, self.global_step)

                # Save checkpoint
                if self.global_step % self.save_every == 0:
                    self._save_checkpoint(f"checkpoint_step_{self.global_step}")

            # Epoch summary
            avg_train_loss = np.mean(epoch_losses)

            # CRITICAL: Verify MTP heads are actually learning (weights changed from init)
            if epoch == 1 or epoch == self.num_epochs // 2 or epoch == self.num_epochs:
                with torch.no_grad():
                    mtp_std_sum = 0.0
                    for head in self.model.mtp_heads:
                        for param in head.parameters():
                            if len(param.shape) >= 2:  # Weight matrices only
                                mtp_std_sum += param.std().item()
                                break
                    avg_mtp_std = mtp_std_sum / len(self.model.mtp_heads)
                    self.logger.info(f"  MTP weight avg std: {avg_mtp_std:.6f} (init was ~0.02)")
                    if abs(avg_mtp_std - 0.02) < 0.001:
                        self.logger.warning("  ⚠️  MTP heads still at initialization! Training may not be working.")
                    else:
                        self.logger.info(f"  ✓ MTP heads have changed from init by {(avg_mtp_std - 0.02):.6f}")

            # Calculate component loss averages for the epoch
            avg_ce_loss = np.mean(epoch_ce_losses) if epoch_ce_losses else 0.0
            avg_mse_loss = np.mean(epoch_mse_losses) if epoch_mse_losses else 0.0

            # Validation
            val_loss = self._validation_step(epoch=epoch + 1)

            epoch_stat = {
                "epoch": epoch + 1,
                "avg_train_loss": avg_train_loss,
                "avg_val_loss": val_loss,
                "avg_ce_loss": avg_ce_loss,
                "avg_mse_loss": avg_mse_loss,
                "per_head_avg_loss": {},
                "active_heads": active_heads,
                "gpu_memory": self.gpu_monitor.get_memory_stats()
            }
            for i in range(self.model.speculation_depth):
                if epoch_mtp_losses[i]:
                    epoch_stat["per_head_avg_loss"][f"head_{i+1}"] = np.mean(epoch_mtp_losses[i])

            epoch_stats.append(epoch_stat)

            # Log to TensorBoard
            self.writer.add_scalar("epoch/train_loss", avg_train_loss, epoch + 1)
            if val_loss is not None:
                self.writer.add_scalar("epoch/val_loss", val_loss, epoch + 1)

            self.logger.info(f"Epoch {epoch + 1} summary:")
            self.logger.info(f"  Train loss: {avg_train_loss:.6f} (CE: {avg_ce_loss:.6f}, MSE: {avg_mse_loss:.6f})")
            if val_loss is not None:
                self.logger.info(f"  Val loss:   {val_loss:.6f}")
                # Check for overfitting
                if val_loss > avg_train_loss * 1.5:
                    self.logger.warning(f"  Possible overfitting detected (val >> train)")

            # Log per-head losses with clearer formatting
            if epoch_stat["per_head_avg_loss"]:
                self.logger.info(f"  Per-head MSE losses (curriculum: heads 1-{active_heads} trained):")
                for head, loss_val in sorted(epoch_stat["per_head_avg_loss"].items()):
                    status = "trained" if int(head.split("_")[1]) <= active_heads else "inactive"
                    self.logger.info(f"    {head}: {loss_val:.6f} ({status})")

            # Log memory after epoch
            self.gpu_monitor.log_memory_summary()

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
                self.logger.info(f"  *** New best model saved! ***")
            else:
                patience_counter += 1
                if patience_counter >= patience and self.val_loader is not None:
                    self.logger.warning(f"  Early stopping triggered (patience={patience})")
                    break

        # Save training history
        history_path = self.output_dir / "training_history.json"

        # Add GPU memory report to training history
        memory_report = self.gpu_monitor.get_memory_report()

        with open(history_path, "w") as f:
            json.dump({
                "final_best_loss": best_val_loss,
                "epochs": epoch_stats,
                "gpu_memory_report": memory_report,
                "config": {
                    "drafter_model": self.drafter_model_name,
                    "num_epochs": self.num_epochs,
                    "batch_size": self.train_loader.batch_size,
                    "learning_rate": self.optimizer.defaults["lr"],
                    "warmup_steps": self.warmup_steps
                }
            }, f, indent=2)
        self.logger.info(f"\nTraining history saved to {history_path}")

        # Log final memory summary
        self.logger.info(f"\n{'='*50}")
        self.logger.info("GPU MEMORY SUMMARY")
        self.logger.info(f"{'='*50}")
        self.gpu_monitor.log_memory_summary()

        self.writer.close()
        self.logger.info("\nTraining complete!")

    def _get_active_heads(self, epoch: int) -> int:
        """Determine how many MTP heads to train based on curriculum schedule.

        Curriculum learning: Start with head 1 only, gradually add deeper heads.
        This prevents the compounding error problem in speculative decoding.

        Schedule:
        - Epochs 1-2: Train only head 1 (foundation)
        - Epochs 3-4: Train heads 1-2
        - Epochs 5-6: Train heads 1-3
        - Epochs 7+: Train all heads
        """
        if epoch <= 2:
            return 1
        elif epoch <= 4:
            return 2
        elif epoch <= 6:
            return 3
        else:
            return self.model.speculation_depth

    @oom_recovery_handler
    def _training_step(self, batch: Dict[str, torch.Tensor], epoch: int = 1,
                       accumulation_step: int = 0, total_accumulation_steps: int = 1) -> Tuple[torch.Tensor, Dict]:
        """Single training step with P-EAGLE aligned loss and curriculum learning.

        Key insight: During inference, drafter's predicted hidden states are converted
        to tokens via the TARGET model's lm_head. So we must train to match the
        token distributions, not just hidden state vectors.

        Uses curriculum learning: early epochs focus on head 1, progressively
        adding deeper heads as shallower heads converge.

        Args:
            batch: Training batch
            epoch: Current epoch number
            accumulation_step: Current accumulation step (0-indexed)
            total_accumulation_steps: Total number of gradient accumulation steps
        """
        # Only zero gradients on first accumulation step
        is_first_step = (accumulation_step == 0)
        is_last_step = (accumulation_step == total_accumulation_steps - 1)

        if is_first_step:
            self.optimizer.zero_grad()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        target_hidden = batch["target_hidden"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        # Forward pass: drafter generates hidden states WITH target_hidden injection
        # EAGLE-3 CRITICAL: Pass target_hidden for concatenation at first layer
        # This aligns drafter's distribution with target model's distribution
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target_hidden=target_hidden,  # NOW PASSED: EAGLE-3 concatenation injection
            is_training=True
        )

        # Determine active heads for curriculum learning
        active_heads = self._get_active_heads(epoch)

        # Compute losses for each MTP head
        ce_losses = []
        mse_losses = []
        token_accs = []
        mtp_losses = []
        head_mask_coverages = []

        for k, pred_hidden in enumerate(outputs["mtp_predictions"]):
            # CURRICULUM LEARNING: Only train active heads
            if k >= active_heads:
                break

            shift = k + 1

            if pred_hidden.shape[1] > 0:
                target_shifted = target_hidden[:, shift:shift + pred_hidden.shape[1]]
                mask_shifted = loss_mask[:, shift:shift + pred_hidden.shape[1]]

                # DIAGNOSTIC: Track mask coverage for this head
                mask_coverage = mask_shifted.sum().item() / (mask_shifted.numel() + 1e-8)
                head_mask_coverages.append(mask_coverage)

                min_len = min(pred_hidden.shape[1], target_shifted.shape[1])

                if min_len > 0:
                    pred_trimmed = pred_hidden[:, :min_len]
                    target_trimmed = target_shifted[:, :min_len]
                    mask_trimmed = mask_shifted[:, :min_len]

                    # Skip if mask is all zeros (no learning signal)
                    if mask_trimmed.sum() == 0:
                        continue

                    # P-EAGLE aligned loss: match token distributions via target lm_head
                    # Uses cross-entropy on hard targets for stable gradients
                    ce_loss_k, mse_loss_k, acc_k = hidden_state_token_loss(
                        pred_trimmed,
                        target_trimmed,
                        self.target_lm_head,
                        mask_trimmed,
                        temperature=1.0,
                        ce_weight=1.0,
                        mse_weight=0.1
                    )

                    ce_losses.append(ce_loss_k)
                    mse_losses.append(mse_loss_k)
                    token_accs.append(acc_k.item())
                    # Store raw MSE for reporting (not scaled)
                    mtp_losses.append(mse_loss_k.item())

        # Combine losses: Cross-Entropy (token matching) + MSE (hidden state)
        # CE ensures tokens match, MSE ensures hidden states are structurally similar
        total_loss = torch.tensor(0.0, device=self.device)

        if ce_losses:
            # Weight later heads less - they depend on earlier heads being accurate
            weighted_ce = []
            for i, ce in enumerate(ce_losses):
                # Head 1: 1.0, Head 2: 0.9, Head 3: 0.8, etc.
                weight = max(0.5, 1.0 - i * 0.1)
                weighted_ce.append(ce * weight)
            ce_total = sum(weighted_ce) / sum(max(0.5, 1.0 - i * 0.1) for i in range(len(weighted_ce)))
            total_loss = total_loss + ce_total  # Primary: token matching

        if mse_losses:
            mse_total = sum(mse_losses) / len(mse_losses)
            total_loss = total_loss + 0.1 * mse_total  # Secondary: hidden state similarity

        # --- LOSS SCALING ---
        # The CE loss is already properly scaled in hidden_state_token_loss.
        # We apply a small multiplier to ensure healthy gradient magnitudes.
        # CE loss is the primary driver - it ensures token predictions match.
        # MSE is secondary - it keeps hidden states structurally similar.
        if total_loss.item() > 0:
            total_loss = total_loss * 10.0  # Scale up for healthy gradients
        # --------------------------

        if total_loss.item() == 0:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Scale loss for gradient accumulation
        if total_accumulation_steps > 1:
            total_loss = total_loss / total_accumulation_steps

        # Backward pass
        total_loss.backward()

        # CRITICAL FIX: Log gradient norms before clipping to diagnose issues
        if self.global_step % 10 == 0:
            mtp_grad_norm = 0.0
            lora_grad_norm = 0.0
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    if "mtp_heads" in name:
                        mtp_grad_norm += param.grad.norm().item() ** 2
                    elif "lora" in name.lower():
                        lora_grad_norm += param.grad.norm().item() ** 2
            if mtp_grad_norm > 0:
                self.writer.add_scalar("gradients/mtp_norm", mtp_grad_norm ** 0.5, self.global_step)
            if lora_grad_norm > 0:
                self.writer.add_scalar("gradients/lora_norm", lora_grad_norm ** 0.5, self.global_step)

        # Gradient clipping per parameter group (less aggressive for MTP heads)
        # LoRA: clip at max_grad_norm (1.0 default)
        # MTP heads: clip at max_grad_norm * 5 (5.0) - allow larger updates
        for group in self.optimizer.param_groups:
            if group.get('lr', 0) > self.optimizer.defaults.get('lr', 2e-5) * 2:
                # High LR group = MTP heads, allow larger gradients
                torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm * 5)
            else:
                torch.nn.utils.clip_grad_norm_(group['params'], self.max_grad_norm)

        # Only update weights on last accumulation step
        if is_last_step:
            self.optimizer.step()
            self.scheduler.step()

        metrics = {
            "mtp_loss_avg": np.mean(mtp_losses) if mtp_losses else 0.0,
            "ce_loss_avg": sum(ce.item() for ce in ce_losses) / len(ce_losses) if ce_losses else 0.0,
            "mse_loss_avg": sum(mse_losses) / len(mse_losses) if mse_losses else 0.0,
            "token_acc_avg": np.mean(token_accs) if token_accs else 0.0,
            "active_heads": active_heads,
            "avg_mask_coverage": np.mean(head_mask_coverages) if head_mask_coverages else 0.0,
        }
        for i, loss_i in enumerate(mtp_losses):
            metrics[f"mtp_loss_{i+1}"] = loss_i
        for i, acc_i in enumerate(token_accs):
            metrics[f"token_acc_{i+1}"] = acc_i

        return total_loss, metrics

    def _save_checkpoint(self, name: str):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return

        checkpoint_dir = self.output_dir / name
        # Unwrap DDP model if needed before saving
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        # Pass target_lm_head to ensure vocab compatibility during inference
        model_to_save.save_checkpoint(str(checkpoint_dir), target_lm_head=self.target_lm_head)
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Drafter Training")
    parser.add_argument("--drafter_model", required=True, help="Base model for drafter")
    parser.add_argument("--target_hidden_dim", type=int, required=True)
    parser.add_argument("--speculation_depth", type=int, default=4)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--use_lora", action="store_true", default=False)
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
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom name for this training run (used in logs)")
    parser.add_argument("--gpu-safety-margin", type=float, default=1.5,
                        help="Minimum GPU memory to keep free in GB (default: 1.5)")
    parser.add_argument("--yes", action="store_true",
                        help="Skip confirmation prompt and start training immediately")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint directory (e.g., checkpoints_peagle_v2/checkpoint_step_1000)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps to accumulate gradients before updating weights. Larger values = less memory but slower training (default: 1)")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length for training. Reduce to 1024 to save VRAM on memory-constrained systems (default: 2048)")

    args = parser.parse_args()

    # Initialize distributed training if using multiple GPUs
    rank, world_size, local_rank = setup_distributed()
    is_main = (rank == 0)

    # Setup logging FIRST - before anything else (only on main process for file logging)
    output_path = Path(args.output_dir)
    if is_main:
        output_path.mkdir(parents=True, exist_ok=True)
    # Barrier to ensure directory is created before other processes proceed
    if world_size > 1:
        dist.barrier()
    logger, run_log_dir, run_id = setup_training_logger(output_path, args.run_name)

    # Log all arguments
    logger.info("Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  --{arg}: {value}")
    logger.info("")

    # Run pre-training security verification
    if args.dataset_source:
        if not verify_dataset_source_security(args.dataset_source, args.skip_security_check):
            logger.error("\n⛔ Training aborted due to security concerns.")
            logger.error("   Use --skip-security-check to override (not recommended)")
            exit(1)
    else:
        # Check feature directory security
        if not run_pre_training_security_check(args.feature_dir):
            logger.error("\n⛔ Training aborted due to security concerns.")
            exit(1)

    # Save configuration to log directory
    config_path = run_log_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")

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
        yes=args.yes,
        quantization=args.quantization,
        logger=logger,
        run_log_dir=run_log_dir,
        gpu_safety_margin_gb=args.gpu_safety_margin,
        resume_from=args.resume,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_len=args.max_seq_len,
        rank=rank,
        world_size=world_size
    )

    try:
        trainer.train()
        if is_main:
            logger.info(f"\n{'='*70}")
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*70}")
            logger.info(f"Logs saved to: {run_log_dir}")
            logger.info(f"Best model: {args.output_dir}/best_model")
    except Exception as e:
        if is_main:
            logger.exception("Training failed with error:")
            logger.error(f"\n{'='*70}")
            logger.error("TRAINING FAILED")
            logger.error(f"{'='*70}")
            logger.error(f"See logs for details: {run_log_dir}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
