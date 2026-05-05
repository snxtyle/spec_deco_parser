#!/usr/bin/env python3
"""
Pre-flight check for P-EAGLE training pipeline.

Validates each stage BEFORE spending hours on training:
1. Dataset format
2. Feature extraction compatibility
3. Model/tokenizer alignment
4. Training configuration
"""

import argparse
import json
import sys
from pathlib import Path


def check_stage1_dataset(dataset_path: str) -> tuple[bool, dict]:
    """Check dataset is valid for feature extraction."""
    print("\n" + "="*60)
    print("STAGE 1: Dataset Validation")
    print("="*60)

    try:
        with open(dataset_path, 'r') as f:
            first_line = f.readline()
            sample = json.loads(first_line)
    except Exception as e:
        return False, {"error": f"Cannot read dataset: {e}"}

    # Check required fields
    checks = {
        "has_messages": "messages" in sample,
        "has_loss_mask_segments": "loss_mask_segments" in sample,
        "has_train_indices": "train_indices" in sample.get("loss_mask_segments", {}),
        "has_nonzero_train_indices": len(sample.get("loss_mask_segments", {}).get("train_indices", [])) > 0
    }

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")

    all_passed = all(checks.values())

    if all_passed:
        train_count = len(sample["loss_mask_segments"]["train_indices"])
        msg_count = len(sample["messages"])
        print(f"\n  Sample stats:")
        print(f"    - Messages: {msg_count}")
        print(f"    - Trainable messages: {train_count}")

    return all_passed, checks


def check_stage2_feature_compatibility(
    dataset_path: str,
    target_model: str,
    drafter_model: str
) -> tuple[bool, dict]:
    """Check models and tokenizers are compatible."""
    print("\n" + "="*60)
    print("STAGE 2: Feature Extraction Compatibility")
    print("="*60)

    try:
        from transformers import AutoTokenizer
        import torch
    except ImportError as e:
        return False, {"error": f"Missing dependency: {e}"}

    results = {}

    # Check tokenizers load
    print(f"\n  Loading tokenizers...")
    try:
        target_tok = AutoTokenizer.from_pretrained(target_model)
        results["target_tokenizer"] = True
        print(f"  ✅ Target tokenizer ({target_model}): vocab={len(target_tok)}")
    except Exception as e:
        results["target_tokenizer"] = False
        print(f"  ❌ Target tokenizer failed: {e}")
        return False, results

    try:
        drafter_tok = AutoTokenizer.from_pretrained(drafter_model)
        results["drafter_tokenizer"] = True
        print(f"  ✅ Drafter tokenizer ({drafter_model}): vocab={len(drafter_tok)}")
    except Exception as e:
        results["drafter_tokenizer"] = False
        print(f"  ❌ Drafter tokenizer failed: {e}")
        return False, results

    # Check vocab compatibility
    print(f"\n  Checking vocab compatibility...")
    test_phrases = ["hello world", "What's the weather?", "The answer is 42."]

    mismatches = 0
    for phrase in test_phrases:
        target_ids = target_tok.encode(phrase, add_special_tokens=False)
        drafter_ids = drafter_tok.encode(phrase, add_special_tokens=False)
        if target_ids != drafter_ids:
            mismatches += 1

    results["vocab_compatible"] = mismatches == 0
    if mismatches == 0:
        print(f"  ✅ Tokenizers produce identical encodings")
    else:
        print(f"  ⚠️  {mismatches}/{len(test_phrases)} test phrases encode differently")
        print(f"     Will use drafter tokenizer during training (handled automatically)")

    # Test tokenization with actual sample
    print(f"\n  Testing dataset tokenization...")
    try:
        with open(dataset_path, 'r') as f:
            sample = json.loads(f.readline())

        messages = sample["messages"]
        conversation_text = "\n".join([
            f"<{m['role']}>{m.get('content', '')}</{m['role']}>"
            for m in messages if m.get('content')
        ])

        tokens = drafter_tok(conversation_text, max_length=2048, truncation=True)
        token_count = len(tokens["input_ids"])

        results["tokenization_works"] = True
        print(f"  ✅ Sample tokenizes successfully ({token_count} tokens)")

        if token_count >= 2048:
            print(f"  ⚠️  Sample is at max_length (truncation will occur)")

    except Exception as e:
        results["tokenization_works"] = False
        print(f"  ❌ Tokenization failed: {e}")

    all_passed = all(results.values())
    return all_passed, results


def check_stage3_training_config(
    feature_dir: str,
    target_hidden_dim: int,
    drafter_model: str
) -> tuple[bool, dict]:
    """Check training configuration is valid."""
    print("\n" + "="*60)
    print("STAGE 3: Training Configuration")
    print("="*60)

    results = {}

    # Check feature files exist
    print(f"\n  Checking feature directory: {feature_dir}")
    feature_path = Path(feature_dir)

    if not feature_path.exists():
        print(f"  ⚠️  Feature dir doesn't exist yet (will be created)")
        results["features_exist"] = None
    else:
        feature_files = list(feature_path.glob("*_shard*.pt"))
        if feature_files:
            print(f"  ✅ Found {len(feature_files)} feature files")
            results["features_exist"] = True

            # Check one feature file
            import torch
            sample_file = feature_files[0]
            try:
                data = torch.load(sample_file, map_location="cpu")
                print(f"  ✅ Feature file loads successfully: {sample_file.name}")
                print(f"     Samples: {data.get('num_samples', '?')}")
                print(f"     Hidden size: {data.get('hidden_size', '?')}")
                print(f"     Vocab size: {data.get('vocab_size', '?')}")

                # Check hidden dim matches
                if "hidden_size" in data:
                    if data["hidden_size"] == target_hidden_dim:
                        print(f"  ✅ Hidden dim matches target ({target_hidden_dim})")
                        results["hidden_dim_match"] = True
                    else:
                        print(f"  ❌ Hidden dim mismatch: feature={data['hidden_size']}, config={target_hidden_dim}")
                        results["hidden_dim_match"] = False

            except Exception as e:
                print(f"  ❌ Failed to load feature file: {e}")
                results["features_valid"] = False
        else:
            print(f"  ⚠️  No feature files found (will be generated)")
            results["features_exist"] = None

    # Check drafter model config
    print(f"\n  Checking drafter model: {drafter_model}")
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(drafter_model)
        print(f"  ✅ Drafter config loads")
        print(f"     Hidden size: {config.hidden_size}")
        print(f"     Num layers: {config.num_hidden_layers}")
        results["drafter_config"] = True
    except Exception as e:
        print(f"  ❌ Failed to load drafter config: {e}")
        results["drafter_config"] = False

    all_passed = all(v for v in results.values() if v is not None)
    return all_passed, results


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check for P-EAGLE")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL")
    parser.add_argument("--target-model", default="google/gemma-7b-it")
    parser.add_argument("--drafter-model", default="google/gemma-2b-it")
    parser.add_argument("--target-hidden-dim", type=int, default=3072)
    parser.add_argument("--feature-dir", default="data/features")
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")

    args = parser.parse_args()

    print("\n" + "="*60)
    print("P-EAGLE PRE-FLIGHT CHECK")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Target: {args.target_model}")
    print(f"Drafter: {args.drafter_model}")

    all_passed = True

    # Stage 1
    if args.stage in ["1", "all"]:
        passed, _ = check_stage1_dataset(args.dataset)
        all_passed = all_passed and passed

    # Stage 2
    if args.stage in ["2", "all"]:
        passed, _ = check_stage2_feature_compatibility(
            args.dataset, args.target_model, args.drafter_model
        )
        all_passed = all_passed and passed

    # Stage 3
    if args.stage in ["3", "all"]:
        passed, _ = check_stage3_training_config(
            args.feature_dir, args.target_hidden_dim, args.drafter_model
        )
        all_passed = all_passed and passed

    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("🚀 READY FOR LAUNCH - All checks passed!")
    else:
        print("⛔ ABORT - Fix issues before proceeding")
    print("="*60 + "\n")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
