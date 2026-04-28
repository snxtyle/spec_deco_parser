#!/usr/bin/env python3
"""
Feature Extraction Utilities for P-EAGLE
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Tuple
from transformers import PreTrainedTokenizer


class EagleDataset(Dataset):
    """
    Dataset for P-EAGLE feature extraction.
    Reads JSONL with format:
    {
        "messages": [{"role": "system", "content": "..."}, ...],
        "loss_mask_segments": {
            "train_indices": [2, 4],
            "ignore_indices": [0, 1, 3],
            "segments": [{"index": 0, "role": "system", "mask": 0}, ...]
        }
    }
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.samples = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        messages = sample["messages"]
        conversation_text = self._messages_to_text(messages)
        loss_mask_segments = sample.get("loss_mask_segments", {})
        segments = loss_mask_segments.get("segments", [])

        return {
            "conversation_text": conversation_text,
            "segments": segments,
            "original_messages": messages
        }

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI message format to text."""
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if content:
                parts.append(f"<{role}>{content}</{role}>")
        return "\n".join(parts)


class EagleTrainingDataset(Dataset):
    """Dataset for training the P-EAGLE drafter.

    Works with ANY model pair: features from any target model can train
    any drafter model. Handles tokenizer mismatch automatically.
    """

    def __init__(
        self,
        feature_dir: str,
        tokenizer: PreTrainedTokenizer,
        speculation_depth: int = 4,
        max_seq_len: int = 2048
    ):
        self.tokenizer = tokenizer
        self.speculation_depth = speculation_depth
        self.max_seq_len = max_seq_len

        self.samples = []
        from pathlib import Path

        for pt_file in sorted(Path(feature_dir).glob("*_shard*.pt")):
            print(f"Loading {pt_file}")
            data = torch.load(pt_file, map_location="cpu")
            num_samples = data["num_samples"]

            # Get texts list if available (new format)
            texts = data.get("texts", [None] * num_samples)

            for i in range(num_samples):
                self.samples.append({
                    "text": texts[i] if i < len(texts) else None,
                    "input_ids": data["input_ids"][i],
                    "fused_hidden_states": data["fused_hidden_states"][i],
                    "loss_mask": data["loss_mask"][i],
                    "attention_mask": data["attention_mask"][i]
                })

        print(f"Loaded {len(self.samples)} training samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Anchor to target's hidden state length (this is ground truth)
        target_len = min(len(sample["fused_hidden_states"]), self.max_seq_len)

        # Retokenize with drafter's tokenizer if text available
        if sample["text"] is not None:
            drafter_inputs = self.tokenizer(
                sample["text"],
                return_tensors="pt",
                max_length=self.max_seq_len,
                truncation=True,
                padding=False
            )
            drafter_ids = drafter_inputs["input_ids"][0]
            drafter_mask = drafter_inputs["attention_mask"][0]

            # Align drafter sequence to target length
            if len(drafter_ids) >= target_len:
                input_ids = drafter_ids[:target_len]
                attention_mask = drafter_mask[:target_len]
            else:
                # Pad to match target length
                pad_len = target_len - len(drafter_ids)
                pad_id = self.tokenizer.pad_token_id or 0
                input_ids = torch.cat([drafter_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                attention_mask = torch.cat([drafter_mask, torch.zeros(pad_len, dtype=torch.long)])
        else:
            # Fallback: use target's tokenization (may cause vocab mismatch)
            input_ids = sample["input_ids"][:target_len]
            attention_mask = sample["attention_mask"][:target_len]

        # Target hidden states are used for both:
        # 1. Hidden state injection during forward pass (if enabled)
        # 2. Ground truth for MTP loss calculation
        target_hidden = sample["fused_hidden_states"][:target_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_hidden": target_hidden,
            "loss_mask": sample["loss_mask"][:target_len]
        }


def align_segments_to_tokens(
    messages: List[Dict[str, str]],
    segments: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor
) -> torch.Tensor:
    """
    Align logical segment masks (per-message) to token-level masks.

    FIXED: Properly handles various content types and uses same tokenization
    as conversation_text construction to ensure position alignment.
    """
    seq_len = input_ids.shape[0]
    token_mask = torch.zeros(seq_len, dtype=torch.int32)

    # Build conversation text same way as _messages_to_text
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle various content types
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, (list, dict)):
            # Convert to JSON string for consistent tokenization
            import json
            content_str = json.dumps(content)
        else:
            content_str = str(content) if content else ""

        if content_str:
            parts.append(f"<{role}>{content_str}</{role}>")

    # Tokenize the full conversation to match input_ids
    full_text = "\n".join(parts)
    full_tokens = tokenizer.encode(full_text, add_special_tokens=True)

    # If lengths don't match, something is wrong - fallback to heuristic
    if len(full_tokens) != seq_len:
        # Fallback: mark all non-padding tokens after first user message
        print(f"  Warning: Token length mismatch ({len(full_tokens)} vs {seq_len}), using heuristic mask")
        # Mark positions from middle to end as trainable (rough heuristic)
        start_pos = seq_len // 4  # Start from 25% into sequence
        token_mask[start_pos:] = 1
        # Respect attention mask
        if input_ids is not None:
            # Don't train on padding
            non_pad = (input_ids != tokenizer.pad_token_id).int()
            token_mask = token_mask * non_pad
        return token_mask

    # Build accurate token ranges by tracking position in full token sequence
    current_pos = 0
    token_ranges = []

    for msg_idx, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Handle various content types same as above
        if isinstance(content, str):
            content_str = content
        elif isinstance(content, (list, dict)):
            import json
            content_str = json.dumps(content)
        else:
            content_str = str(content) if content else ""

        if content_str:
            # Calculate positions for JUST the content (not the <role> tags)
            # <role>CONTENT</role>
            # ^      ^      ^
            # |      |      |
            # start content end
            opening_tag = f"<{role}>"
            closing_tag = f"</{role}>"

            opening_tokens = tokenizer.encode(opening_tag, add_special_tokens=False)
            content_tokens = tokenizer.encode(content_str, add_special_tokens=False)
            closing_tokens = tokenizer.encode(closing_tag, add_special_tokens=False)

            wrapper_len = len(opening_tokens) + len(content_tokens) + len(closing_tokens)

            # Content starts after opening tag
            content_start = current_pos + len(opening_tokens)
            content_end = min(content_start + len(content_tokens), seq_len)

            # Store both wrapper range (for tracking) and content range (for masking)
            token_ranges.append((content_start, content_end, msg_idx))
            current_pos += wrapper_len
        else:
            token_ranges.append((current_pos, current_pos, msg_idx))

    # Apply masks from segments
    for seg in segments:
        msg_idx = seg.get("index", -1)
        mask_value = seg.get("mask", 0)

        if msg_idx < 0:
            continue

        for start, end, idx in token_ranges:
            if idx == msg_idx:
                if mask_value == 1:
                    token_mask[start:end] = 1
                break

    # Safety: ensure mask doesn't exceed sequence bounds
    token_mask = token_mask[:seq_len]

    return token_mask


def fuse_tri_layer_features(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_indices: List[int],
    fusion_mode: str = "mean"
) -> torch.Tensor:
    """
    Fuse hidden states from multiple layers into a single vector.
    """
    selected = [hidden_states[i] for i in layer_indices]

    if fusion_mode == "mean":
        fused = torch.stack(selected, dim=0).mean(dim=0)
    elif fusion_mode == "weighted":
        weights = torch.softmax(torch.randn(len(selected)), dim=0)
        fused = sum(w * h for w, h in zip(weights, selected))
    elif fusion_mode == "concat":
        fused = torch.cat(selected, dim=-1)
    else:
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    return fused
