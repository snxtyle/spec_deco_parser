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

    Handles arbitrary model pairs: Features from any target model can train
    any drafter model. Uses unified sequence length based on hidden states.
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

            for i in range(num_samples):
                self.samples.append({
                    "text": data.get("text", None),
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

        # Unified sequence length: use target hidden states as the anchor
        # The drafter must produce predictions aligned to target positions
        target_len = min(len(sample["fused_hidden_states"]), self.max_seq_len)

        # Get text and retokenize with drafter's tokenizer
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

            # Align drafter tokens to target length by truncating/padding
            if len(drafter_ids) >= target_len:
                input_ids = drafter_ids[:target_len]
                attention_mask = drafter_mask[:target_len]
            else:
                # Pad if drafter tokenization is shorter
                pad_len = target_len - len(drafter_ids)
                input_ids = torch.cat([
                    drafter_ids,
                    torch.full((pad_len,), self.tokenizer.pad_token_id or 0, dtype=torch.long)
                ])
                attention_mask = torch.cat([
                    drafter_mask,
                    torch.zeros(pad_len, dtype=torch.long)
                ])
        else:
            # Fallback: use target's input_ids (risks vocab mismatch but maintains length)
            input_ids = sample["input_ids"][:target_len]
            attention_mask = sample["attention_mask"][:target_len]

        return {
            "input_ids": input_ids,
            "target_hidden": sample["fused_hidden_states"][:target_len],
            "loss_mask": sample["loss_mask"][:target_len],
            "attention_mask": attention_mask
        }


def align_segments_to_tokens(
    messages: List[Dict[str, str]],
    segments: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor
) -> torch.Tensor:
    """
    Align logical segment masks (per-message) to token-level masks.
    """
    seq_len = input_ids.shape[0]
    token_mask = torch.zeros(seq_len, dtype=torch.int32)

    token_ranges = []
    start_idx = 0

    for msg_idx, msg in enumerate(messages):
        content = msg.get("content", "")
        if content:
            msg_tokens = tokenizer.encode(content, add_special_tokens=False)
            end_idx = start_idx + len(msg_tokens)
            token_ranges.append((start_idx, end_idx, msg_idx))
            start_idx = end_idx
        else:
            token_ranges.append((start_idx, start_idx, msg_idx))

    for seg in segments:
        msg_idx = seg["index"]
        mask_value = seg["mask"]

        for start, end, idx in token_ranges:
            if idx == msg_idx:
                if mask_value == 1:
                    token_mask[start:end] = 1
                break

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
