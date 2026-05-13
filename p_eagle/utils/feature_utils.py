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
        # Support both old format ('messages') and new format ('original_messages')
        messages = sample.get("messages") or sample.get("original_messages", [])

        # Get segments - handle multiple possible formats:
        # 1. Root level "segments" (current generate_data.py output)
        # 2. Nested "loss_mask_segments.segments" (older format)
        # 3. Auto-generate from message roles if neither exists
        segments = sample.get("segments", [])

        if not segments:
            # Try older nested format
            loss_mask_segments = sample.get("loss_mask_segments", {})
            segments = loss_mask_segments.get("segments", [])

            if not segments and "train_indices" in loss_mask_segments:
                # Convert train_indices format to segments
                train_indices = set(loss_mask_segments.get("train_indices", []))
                for i in range(len(messages)):
                    segments.append({
                        "index": i,
                        "mask": 1 if i in train_indices else 0
                    })

        if not segments and messages:
            # Auto-generate segments from message roles
            # Train on assistant responses (mask=1), ignore system/user (mask=0)
            assistant_roles = {"assistant", "bot", "ai"}
            segments = []
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown").lower().strip()
                content = msg.get("content", "")
                # Assistant with content = train (mask=1), otherwise ignore (mask=0)
                is_assistant = role in assistant_roles
                has_content = bool(content and content.strip())
                mask = 1 if (is_assistant and has_content) else 0
                segments.append({"index": i, "role": role, "mask": mask})

        # CRITICAL FIX: Pre-process messages to serialize tool_calls into content
        # This ensures tool_calls are included in the tokenized text
        # BUT do this AFTER extracting segments to preserve alignment
        processed_messages = []
        for msg in messages:
            msg_copy = msg.copy()
            content = msg_copy.get("content", "")
            if not content and msg_copy.get("tool_calls"):
                # Serialize tool_calls into content so it gets tokenized
                import json
                tool_calls = msg_copy.get("tool_calls", [])
                tool_strs = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                    tool_strs.append(f"{name}({args})")
                msg_copy["content"] = "[TOOL_CALLS]" + "; ".join(tool_strs) + "[/TOOL_CALLS]"
            processed_messages.append(msg_copy)

        # Use processed messages that include serialized tool_calls
        messages = processed_messages
        conversation_text = self._messages_to_text(messages)

        # Validate that at least one segment has mask=1 for training
        if segments and not any(seg.get("mask") == 1 for seg in segments):
            print(f"WARNING: Sample {idx} has no training segments (mask=1). Check your role strings!")

        return {
            "conversation_text": conversation_text,
            "segments": segments,
            "original_messages": messages
        }

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI message format to text. Preserves ALL messages for real-world data."""
        # For LiteLLM logs and real-world data: keep ALL messages including system/tool
        # Just format them in a simple consistent way
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Handle content that might be None or empty
            if content is None:
                content = ""
            elif not isinstance(content, str):
                # Handle list-type content (e.g., from tool calls)
                if isinstance(content, list):
                    content_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content_parts.append(item.get("text", ""))
                    content = "\n".join(content_parts)
                else:
                    content = str(content)

            # Include tool_calls info if present
            tool_calls = msg.get("tool_calls")
            if tool_calls and not content:
                # Format tool calls as content
                tool_strs = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                    tool_strs.append(f"{name}({args})")
                content = "[TOOL_CALLS] " + "; ".join(tool_strs)

            if content:  # Only add if there's content
                parts.append(f"{role}: {content}")

        return "\n\n".join(parts)


class EagleTrainingDataset(Dataset):
    """Memory-efficient dataset for P-EAGLE training.

    Following official EAGLE implementation pattern:
    - Store shard paths, not actual data
    - Load samples on-demand in __getitem__
    - Only current shard is in RAM (~6GB instead of 148GB)

    Works with ANY model pair: features from any target model can train
    any drafter model. Handles tokenizer mismatch automatically.
    """

    def __init__(
        self,
        feature_dir: str,
        tokenizer: PreTrainedTokenizer,
        speculation_depth: int = 4,
        max_seq_len: int = 2048,
        shard_cache_size: int = 1  # Number of shards to keep in RAM
    ):
        self.tokenizer = tokenizer
        self.speculation_depth = speculation_depth
        self.max_seq_len = max_seq_len
        self.shard_cache_size = shard_cache_size

        from pathlib import Path

        # Store shard paths, not data (EAGLE official pattern)
        self.shard_files = sorted(Path(feature_dir).glob("*_shard*.pt"))
        if not self.shard_files:
            raise ValueError(f"No feature shards found in {feature_dir}")

        # Build index: global_idx -> (shard_idx, local_idx)
        # This allows efficient on-demand loading
        self.sample_index = []  # List of (shard_idx, local_idx) tuples
        self.shard_sample_counts = []  # Number of samples per shard

        print(f"Indexing {len(self.shard_files)} feature shards...")
        for shard_idx, pt_file in enumerate(self.shard_files):
            # Only load metadata (num_samples), not full data
            data = torch.load(pt_file, map_location="cpu", weights_only=False)
            num_samples = data["num_samples"]
            self.shard_sample_counts.append(num_samples)

            # Add index entries for this shard
            for local_idx in range(num_samples):
                self.sample_index.append((shard_idx, local_idx))

            # Free memory immediately
            del data
            import gc
            gc.collect()  # Force garbage collection to free RAM

        print(f"Indexed {len(self.sample_index)} total samples across {len(self.shard_files)} shards")
        print(f"RAM usage: ~{self.shard_sample_counts[0] * 6 / 1024:.1f}GB per shard (vs {sum(self.shard_sample_counts) * 6 / 1024:.1f}GB if loading all)")

        # Shard cache: only keep recent shards in memory
        self._shard_cache = {}  # shard_idx -> shard_data
        self._shard_access_order = []  # LRU tracking

    def __len__(self) -> int:
        return len(self.sample_index)

    def _load_shard_if_needed(self, shard_idx: int):
        """Load shard into cache if not already present."""
        if shard_idx in self._shard_cache:
            # Update LRU order
            if shard_idx in self._shard_access_order:
                self._shard_access_order.remove(shard_idx)
            self._shard_access_order.append(shard_idx)
            return

        # Load the shard
        pt_file = self.shard_files[shard_idx]
        data = torch.load(pt_file, map_location="cpu", weights_only=False)

        # Store in cache
        self._shard_cache[shard_idx] = data
        self._shard_access_order.append(shard_idx)

        # Evict oldest shards if cache is full
        while len(self._shard_cache) > self.shard_cache_size:
            oldest_shard = self._shard_access_order.pop(0)
            if oldest_shard in self._shard_cache:
                del self._shard_cache[oldest_shard]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get shard and local indices
        shard_idx, local_idx = self.sample_index[idx]

        # Load shard if needed (on-demand loading)
        self._load_shard_if_needed(shard_idx)

        # Get sample from cache
        shard_data = self._shard_cache[shard_idx]
        texts = shard_data.get("texts", [None] * shard_data["num_samples"])

        sample = {
            "text": texts[local_idx] if local_idx < len(texts) else None,
            "input_ids": shard_data["input_ids"][local_idx],
            "fused_hidden_states": shard_data["fused_hidden_states"][local_idx],
            "loss_mask": shard_data["loss_mask"][local_idx],
            "attention_mask": shard_data["attention_mask"][local_idx]
        }

        # Add raw hidden states if available
        if "raw_hidden_states" in shard_data:
            sample["raw_hidden_states"] = shard_data["raw_hidden_states"][local_idx]

        # STEP 1: INTELLIGENT WINDOW SELECTION on ORIGINAL features
        loss_mask_full = sample["loss_mask"]
        seq_len = len(loss_mask_full)

        if seq_len > self.max_seq_len:
            # Find the best window that maximizes mask coverage
            # Use tensor operations for efficiency (avoid slow .tolist() conversion)
            best_start = 0
            best_count = 0
            step = max(1, (seq_len - self.max_seq_len) // 20)  # At least 20 windows checked
            for start in range(0, seq_len - self.max_seq_len + 1, step):
                end = start + self.max_seq_len
                count = int(loss_mask_full[start:end].sum())
                if count > best_count:
                    best_count = count
                    best_start = start

            # If no mask found in any window, use middle of sequence
            if best_count == 0:
                best_start = (seq_len - self.max_seq_len) // 2

            # Slice all tensors to the same window
            slice_range = slice(best_start, min(best_start + self.max_seq_len, seq_len))
            input_ids = sample["input_ids"][slice_range]
            attention_mask = sample["attention_mask"][slice_range]
            loss_mask = loss_mask_full[slice_range]
            if "raw_hidden_states" in sample:
                target_hidden = sample["raw_hidden_states"][slice_range]
            else:
                target_hidden = sample["fused_hidden_states"][slice_range]
        else:
            input_ids = sample["input_ids"]
            attention_mask = sample["attention_mask"]
            loss_mask = loss_mask_full
            if "raw_hidden_states" in sample:
                target_hidden = sample["raw_hidden_states"]
            else:
                target_hidden = sample["fused_hidden_states"]

        # STEP 2: Truncate/pad to exact max_seq_len
        target_len = min(len(target_hidden), self.max_seq_len)

        # Use target's tokenization (already aligned with hidden states)
        # Skip retokenization to maintain alignment with windowed data
        input_ids = input_ids[:target_len]
        attention_mask = attention_mask[:target_len]
        loss_mask_sliced = loss_mask[:target_len]
        target_hidden = target_hidden[:target_len]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_hidden": target_hidden,
            "loss_mask": loss_mask_sliced
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


def _fuzzy_find(text: str, pattern: str, start_pos: int = 0, min_similarity: float = 0.85) -> int:
    """Find pattern in text with fuzzy matching.

    Uses sliding window with character-level similarity to find best match.
    Returns start position of best match, or -1 if no match above threshold.
    """
    if not pattern:
        return -1

    pattern_len = len(pattern)
    text_len = len(text)

    if start_pos + pattern_len > text_len:
        return -1

    best_pos = -1
    best_score = 0.0

    # Slide window and compute similarity
    for i in range(start_pos, text_len - pattern_len + 1):
        window = text[i:i + pattern_len]
        # Simple character overlap similarity
        matches = sum(a == b for a, b in zip(window, pattern))
        score = matches / max(len(window), len(pattern))

        if score > best_score:
            best_score = score
            best_pos = i

    return best_pos if best_score >= min_similarity else -1


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
    FIXED: Re-encode original text to get correct offsets (avoid double tokenization).
    This avoids the "shift bug" from manual index summation.
    """
    seq_len = input_ids.shape[0]
    token_mask = torch.zeros(seq_len, dtype=torch.int32)

    # Decode input_ids to get the full text
    full_text_raw = tokenizer.decode(input_ids, skip_special_tokens=False)
    # Normalize for matching (we'll search in normalized space but map back to raw)
    full_text = _normalize_for_matching(full_text_raw)

    # Use offset_mapping for precise character-to-token alignment
    # CRITICAL FIX: Re-encode the decoded text to get offsets that match input_ids
    try:
        encoding = tokenizer(full_text_raw, return_offsets_mapping=True, add_special_tokens=False)
        offsets = encoding['offset_mapping']

        # Ensure offsets match input_ids length
        if len(offsets) != seq_len:
            # Fallback: use role-based heuristic
            # Count assistant messages and estimate positions
            assistant_count = sum(1 for seg in segments if seg.get("mask") == 1)
            if assistant_count > 0:
                # Rough heuristic: assistant content tends to be in latter half
                start_pos = max(seq_len // 3, seq_len - (seq_len // 4))
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

                # Handle tool_calls - if no content but has tool_calls, serialize them
                if not content and messages[msg_idx].get("tool_calls"):
                    import json
                    tool_calls = messages[msg_idx].get("tool_calls", [])
                    tool_parts = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        args = func.get("arguments", "")
                        tool_parts.append(f"{name}({args})")
                    content = "[TOOL_CALLS:" + "; ".join(tool_parts) + "]"

                if not content:
                    continue

                # Normalize content for matching
                content_normalized = _normalize_for_matching(content)
                if not content_normalized:
                    continue

                # Find content in full text, starting from last position
                # This ensures we match the correct occurrence for each message
                start_char = full_text.find(content_normalized, last_search_pos)

                # Fuzzy fallback: if exact match fails, try approximate matching
                if start_char == -1:
                    start_char = _fuzzy_find(full_text, content_normalized, last_search_pos)
                    if start_char == -1:
                        # Silently skip - content not found (happens for special/formatting content)
                        continue

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
    fusion_mode: str = "mean",
    normalize: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Fuse hidden states from multiple layers into a single vector.

    Args:
        hidden_states: Tuple of hidden state tensors from all layers
        layer_indices: Which layer indices to fuse
        fusion_mode: "mean", "weighted", or "concat"
        normalize: Apply LayerNorm to stabilize extreme values (default: True)

    Returns:
        Dictionary containing:
        - 'fused': Normalized fused hidden states
        - 'mean': Per-token mean for denormalization
        - 'std': Per-token std for denormalization
        - 'raw': Original unnormalized fused hidden states (for lm_head compatibility)
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

    # Save raw fused for lm_head compatibility
    raw_fused = fused.clone()

    # CRITICAL FIX: Normalize to prevent extreme values from blowing up loss
    # But save statistics for denormalization during inference
    mean = None
    std = None
    if normalize:
        # Use layernorm for per-sample normalization across hidden dim
        # This keeps values in a reasonable range (~ -3 to +3)
        original_dtype = fused.dtype
        fused_float = fused.float()

        # Compute per-token mean and std across hidden dimension
        mean = fused_float.mean(dim=-1, keepdim=True)
        var = fused_float.var(dim=-1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + 1e-5)

        # Normalize: (x - mean) / std
        fused_normalized = (fused_float - mean) / std

        # Clip extreme outliers (beyond 5 sigma)
        fused_normalized = torch.clamp(fused_normalized, -5.0, 5.0)

        fused = fused_normalized.to(original_dtype)

    return {
        'fused': fused,
        'mean': mean,
        'std': std,
        'raw': raw_fused
    }
