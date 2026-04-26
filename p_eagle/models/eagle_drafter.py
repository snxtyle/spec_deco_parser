#!/usr/bin/env python3
"""
Eagle Drafter Model with Multi-Token Prediction (MTP) Capability
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, List, Optional


def detect_lora_targets(model: nn.Module) -> List[str]:
    """
    Auto-detect LoRA target modules based on model architecture.
    Works for Llama, Mistral, Gemma, Qwen, Phi, and other LLMs.
    """
    target_patterns = []
    module_names = [name for name, _ in model.named_modules()]

    # Common attention patterns
    attention_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj",           # Llama, Mistral, Gemma, Qwen
        "query", "key", "value", "dense",                  # BERT, GPT-2
        "c_attn", "c_proj",                                # GPT-2 alternate
        "attn.q", "attn.k", "attn.v", "attn.o",            # Some variants
    }

    # Common MLP/FFN patterns
    mlp_targets = {
        "gate_proj", "up_proj", "down_proj",               # Llama, Mistral, Gemma
        "fc1", "fc2",                                       # BERT, some GPT
        "c_fc", "c_proj",                                   # GPT-2
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",    # Qwen
    }

    all_targets = attention_targets | mlp_targets

    for target in all_targets:
        # Check if any module name ends with this target
        for name in module_names:
            if name.endswith(f".{target}") or name == target:
                target_patterns.append(target)
                break

    # Remove duplicates while preserving order
    seen = set()
    unique_targets = []
    for t in target_patterns:
        if t not in seen:
            seen.add(t)
            unique_targets.append(t)

    if not unique_targets:
        # Fallback: common default for most decoder LLMs
        return ["q_proj", "v_proj"]

    return unique_targets


class EagleMTPHead(nn.Module):
    """
    Multi-Token Prediction Head for P-EAGLE.
    Predicts the hidden state at position t+k given the hidden state at position t.
    """

    def __init__(
        self,
        hidden_dim: int,
        target_hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.target_hidden_dim = target_hidden_dim

        layers = []
        in_dim = hidden_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim, dtype=dtype),
                nn.LayerNorm(hidden_dim, dtype=dtype),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, target_hidden_dim, dtype=dtype))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class EagleDrafterModel(nn.Module):
    """
    P-EAGLE Drafter with Multi-Token Prediction capability.

    Architecture:
    1. Base LLM (frozen or LoRA-tuned)
    2. Linear projection to match target dimension
    3. K parallel MTP heads for predicting h_{t+1}, ..., h_{t+K}
    """

    def __init__(
        self,
        base_model_name: str,
        target_hidden_dim: int,
        speculation_depth: int = 4,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05,
        device: str = "cuda"
    ):
        super().__init__()

        self.speculation_depth = speculation_depth
        self.target_hidden_dim = target_hidden_dim
        self.device = device

        # Use local cache to avoid re-downloading
        import os
        cache_dir = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "models_cache")
        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading base drafter model: {base_model_name}")
        print(f"Cache directory: {cache_dir}")
        self.config = AutoConfig.from_pretrained(base_model_name, cache_dir=cache_dir)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device} if torch.cuda.is_available() else "cpu",
            cache_dir=cache_dir
        )

        self.hidden_dim = self.config.hidden_size
        print(f"Base model hidden dim: {self.hidden_dim}")
        print(f"Target model hidden dim: {target_hidden_dim}")

        if use_lora:
            print(f"Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")

            # Auto-detect target modules for this architecture
            target_modules = detect_lora_targets(self.base_model)
            print(f"Detected LoRA targets: {target_modules}")

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()

        # Initialize projection and heads with same dtype as base model (bfloat16)
        self.dim_projection = nn.Linear(self.hidden_dim, target_hidden_dim, dtype=torch.bfloat16).to(device)

        self.mtp_heads = nn.ModuleList([
            EagleMTPHead(target_hidden_dim, target_hidden_dim, num_layers=2, dtype=torch.bfloat16).to(device)
            for _ in range(speculation_depth)
        ])

        print(f"Initialized {speculation_depth} parallel MTP heads")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True
    ) -> Dict[str, torch.Tensor]:

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        base_hidden = outputs.hidden_states[-1]
        projected_hidden = self.dim_projection(base_hidden)

        mtp_predictions = []
        for k, mtp_head in enumerate(self.mtp_heads):
            # Trim projected_hidden to match shifted targets: h_t predicts h_{t+k+1}
            # For target at position i, we need prediction at position i-k-1
            # Valid positions for k-th head: [0, seq_len - k - 1]
            trim = k + 1
            pred_input = projected_hidden[:, :-trim] if trim > 0 else projected_hidden
            pred = mtp_head(pred_input)
            mtp_predictions.append(pred)

        return {
            "base_hidden": base_hidden,
            "projected_hidden": projected_hidden,
            "mtp_predictions": mtp_predictions,
            "lm_logits": outputs.logits
        }

    def get_predicted_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_tokens: int = 1
    ) -> torch.Tensor:
        assert 1 <= num_tokens <= self.speculation_depth
        outputs = self.forward(input_ids, attention_mask)
        return outputs["mtp_predictions"][num_tokens - 1]

    def save_checkpoint(self, checkpoint_dir: str):
        """Save model checkpoint."""
        import json
        from pathlib import Path

        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if isinstance(self.base_model, PeftModel):
            self.base_model.save_pretrained(Path(checkpoint_dir) / "lora_weights")

        torch.save({
            "dim_projection": self.dim_projection.state_dict(),
            "mtp_heads": [head.state_dict() for head in self.mtp_heads],
            "speculation_depth": self.speculation_depth,
            "hidden_dim": self.hidden_dim,
            "target_hidden_dim": self.target_hidden_dim
        }, Path(checkpoint_dir) / "eagle_heads.pt")

        config = {
            "base_model": self.base_model.name_or_path if hasattr(self.base_model, 'name_or_path') else "unknown",
            "speculation_depth": self.speculation_depth,
            "hidden_dim": self.hidden_dim,
            "target_hidden_dim": self.target_hidden_dim
        }
        with open(Path(checkpoint_dir) / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load_checkpoint(cls, checkpoint_dir: str, device: str = "cuda"):
        """Load model from checkpoint."""
        import json
        from pathlib import Path

        with open(Path(checkpoint_dir) / "config.json") as f:
            config = json.load(f)

        model = cls(
            base_model_name=config["base_model"],
            target_hidden_dim=config["target_hidden_dim"],
            speculation_depth=config["speculation_depth"],
            use_lora=False,
            device=device
        )

        checkpoint = torch.load(
            Path(checkpoint_dir) / "eagle_heads.pt",
            map_location=device
        )

        model.dim_projection.load_state_dict(checkpoint["dim_projection"])
        for i, head_state in enumerate(checkpoint["mtp_heads"]):
            model.mtp_heads[i].load_state_dict(head_state)

        return model
