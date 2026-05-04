#!/usr/bin/env python3
"""
Fix Dataset - Add loss_mask_segments to existing dataset.

This script reads a dataset that only has 'messages' and adds proper
'loss_mask_segments' for EAGLE training.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path


def generate_loss_mask_segments(messages):
    """
    Generate segment-based loss mask specification for EAGLE training.

    Returns:
        Dict with train_indices, ignore_indices, and per-message specifications
    """
    train_indices = []
    ignore_indices = []
    segments = []

    for idx, msg in enumerate(messages):
        role = msg.get("role", "")

        # Determine mask: 1 = predict (train), 0 = ignore
        if role == "assistant":
            mask = 1
            train_indices.append(idx)
        else:
            # system, user, tool - don't predict
            mask = 0
            ignore_indices.append(idx)

        segments.append({
            "index": idx,
            "role": role,
            "mask": mask
        })

    return {
        "train_indices": train_indices,
        "ignore_indices": ignore_indices,
        "segments": segments
    }


def parse_nested_content(content):
    """Parse content that may be string-encoded JSON list."""
    if content is None:
        return ""
    if isinstance(content, list):
        texts = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and "text" in c:
                    texts.append(c["text"])
                elif "text" in c:
                    texts.append(c["text"])
        return " ".join(texts) if texts else ""
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("[") and ("'type'" in stripped or '"type"' in stripped):
            try:
                import ast
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    texts = []
                    for c in parsed:
                        if isinstance(c, dict) and "text" in c:
                            texts.append(c["text"])
                    return " ".join(texts) if texts else ""
            except:
                pass
        return content
    return str(content)


def fix_sample(sample):
    """Fix a single sample by adding loss_mask_segments and parsing content."""
    messages = sample.get("messages", [])

    # Parse nested content in all messages
    fixed_messages = []
    for msg in messages:
        fixed_msg = dict(msg)
        if "content" in msg:
            fixed_msg["content"] = parse_nested_content(msg.get("content"))
        fixed_messages.append(fixed_msg)

    # Generate loss mask segments
    loss_mask_segments = generate_loss_mask_segments(fixed_messages)

    return {
        "messages": fixed_messages,
        "loss_mask_segments": loss_mask_segments
    }


def main():
    parser = argparse.ArgumentParser(description="Fix dataset by adding loss_mask_segments")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", default=None, help="Output JSONL file (default: auto-generated)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_path.parent / f"dataset_fixed_{timestamp}.jsonl"

    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

    fixed_samples = []
    stats = {"total": 0, "with_assistant": 0, "assistant_messages": 0}

    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)
            stats["total"] += 1

            fixed_sample = fix_sample(sample)
            fixed_samples.append(fixed_sample)

            # Stats
            segments = fixed_sample["loss_mask_segments"]["segments"]
            train_count = sum(1 for s in segments if s["mask"] == 1)
            if train_count > 0:
                stats["with_assistant"] += 1
                stats["assistant_messages"] += train_count

    # Write fixed dataset
    with open(output_path, 'w') as f:
        for sample in fixed_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n=== SUMMARY ===")
    print(f"Total samples: {stats['total']}")
    print(f"Samples with assistant (trainable): {stats['with_assistant']}")
    print(f"Total assistant messages: {stats['assistant_messages']}")
    print(f"Output: {output_path}")

    # Show first sample structure
    if fixed_samples:
        print(f"\n=== SAMPLE STRUCTURE ===")
        sample = fixed_samples[0]
        print(f"Keys: {list(sample.keys())}")
        print(f"Messages count: {len(sample['messages'])}")
        print(f"Segments count: {len(sample['loss_mask_segments']['segments'])}")
        train_indices = sample['loss_mask_segments']['train_indices']
        print(f"Train indices (assistant): {train_indices[:5]}...")


if __name__ == "__main__":
    main()
