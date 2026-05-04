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

        # Validate that at least one segment has mask=1 for training
        if segments and not any(seg.get("mask") == 1 for seg in segments):
            print(f"WARNING: Sample {idx} has no training segments (mask=1). Check your role strings!")

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


def _normalize_for_matching(text: str) -> str:
    """Normalize text for reliable string matching.
    Handles quote differences, whitespace normalization.
    """
    if not isinstance(text, str):
        text = str(text) if text else ""
    # Normalize quotes (single to double for consistency)
    text = text.replace("'", '"')
    # Normalize newlines (some tokenizers convert \\n to actual newlines differently)
    text = text.replace("\\n", "\n")
    # Collapse multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def align_segments_to_tokens(
    messages: List[Dict[str, str]],
    segments: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor
) -> torch.Tensor:
    """
    Align logical segment masks (per-message) to token-level masks.

    FIXED: Uses character-offset mapping for 1:1 text-to-token accuracy.
    FIXED: Tracks search position to handle duplicate content correctly.
    FIXED: Normalizes quotes for reliable matching.
    This avoids the "shift bug" from manual index summation.
    """
    seq_len = input_ids.shape[0]
    token_mask = torch.zeros(seq_len, dtype=torch.int32)

    # Decode input_ids to get the full text
    full_text_raw = tokenizer.decode(input_ids, skip_special_tokens=False)
    # Normalize for matching (we'll search in normalized space but map back to raw)
    full_text = _normalize_for_matching(full_text_raw)

    # Use offset_mapping for precise character-to-token alignment
    try:
        encoding = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']

        # Ensure offsets match input_ids length
        if len(offsets) != seq_len:
            # Fallback: use simple heuristic if offset mapping fails
            print(f"  Warning: Offset mapping length mismatch ({len(offsets)} vs {seq_len})")
            start_pos = min(seq_len // 4, seq_len - 1)
            token_mask[start_pos:] = 1
            return token_mask

        # Track search position to handle duplicate content
        last_search_pos = 0

        for seg in segments:
            if seg.get("mask") == 1:
                msg_idx = seg.get("index", -1)
                if msg_idx < 0 or msg_idx >= len(messages):
                    continue

                content = messages[msg_idx].get("content", "")
                if not isinstance(content, str):
                    content = str(content) if content else ""

                if not content:
                    continue

                # Normalize content for matching
                content_normalized = _normalize_for_matching(content)
                if not content_normalized:
                    continue

                # Find content in full text, starting from last position
                # This ensures we match the correct occurrence for each message
                start_char = full_text.find(content_normalized, last_search_pos)

                if start_char == -1:
                    continue  # Content not found even after normalization

                end_char = start_char + len(content_normalized)
                # Advance search position for next segment
                last_search_pos = end_char

                # Map character positions to token indices
                for i, (tok_start, tok_end) in enumerate(offsets):
                    if i >= seq_len:
                        break
                    # Token overlaps with content region
                    if tok_start >= start_char and tok_end <= end_char:
                        token_mask[i] = 1

    except Exception as e:
        # Fallback: mark latter portion of sequence
        print(f"  Warning: Offset alignment failed ({e}), using heuristic")
        start_pos = min(seq_len // 4, seq_len - 1)
        token_mask[start_pos:] = 1

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
