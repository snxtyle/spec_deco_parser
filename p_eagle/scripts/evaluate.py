#!/usr/bin/env python3
"""
P-EAGLE Comprehensive Evaluation Script
Correct architecture: Drafter predicts hidden states → Target's lm_head → Tokens
"""

import argparse
import json
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

# Disable CUDA graphs and torch.compile to avoid dynamic shape issues
torch._inductor.config.triton.cudagraphs = False
os.environ["TORCHINDUCTOR_CUDAGRAPHS"] = "0"


def evaluate_raw_model(target_model_name: str, prompts: List[str], max_tokens: int = 100) -> Dict:
    """Evaluate baseline (raw target model) performance."""
    print("\n" + "="*70)
    print("  EVALUATING RAW TARGET MODEL (Baseline)")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    results = []
    total_time = 0
    total_tokens = 0

    for i, prompt in enumerate(prompts):
        print(f"  Sample {i+1}/{len(prompts)}...", end=" ")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        start = time.time()

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        elapsed = time.time() - start
        tokens_generated = output.shape[1] - input_ids.shape[1]

        results.append({
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "tokens": tokens_generated,
            "time": elapsed,
            "tps": tokens_generated / elapsed
        })

        total_time += elapsed
        total_tokens += tokens_generated
        print(f"{tokens_generated} tokens, {elapsed:.2f}s, {tokens_generated/elapsed:.1f} tps")

    return {
        "model": target_model_name,
        "total_samples": len(prompts),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "mean_tps": total_tokens / total_time,
        "samples": results
    }


def evaluate_peagle(drafter_checkpoint: str, target_model_name: str,
                    prompts: List[str], max_tokens: int = 100) -> Dict:
    """Evaluate P-EAGLE with correct hidden state prediction."""
    from p_eagle.models.peagle_drafter import EagleDrafterModel

    print("\n" + "="*70)
    print("  EVALUATING P-EAGLE MODEL")
    print("="*70)

    # Load tokenizer and target model
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16
    ).to("cuda")
    target_model.eval()

    # Get target's lm_head (fallback if checkpoint doesn't have saved lm_head)
    if hasattr(target_model, 'lm_head'):
        target_lm_head_fallback = target_model.lm_head
    elif hasattr(target_model, 'model') and hasattr(target_model.model, 'lm_head'):
        target_lm_head_fallback = target_model.model.lm_head
    else:
        raise ValueError("Could not find lm_head in target model")

    # Load drafter
    print(f"Loading drafter from {drafter_checkpoint}...")
    drafter = EagleDrafterModel.load_checkpoint(drafter_checkpoint, device="cuda")
    drafter.eval()

    # Use saved lm_head from checkpoint if available (for vocab compatibility)
    if hasattr(drafter, 'target_lm_head') and drafter.target_lm_head is not None:
        target_lm_head = drafter.target_lm_head
        print(f"Using saved lm_head from checkpoint (vocab: {target_lm_head.weight.shape[0]})")
    else:
        target_lm_head = target_lm_head_fallback
        print(f"Using target model's lm_head (vocab: {target_lm_head.weight.shape[0]})")
        print("WARNING: Vocab mismatch possible if training used different tokenizer")

    speculation_depth = drafter.speculation_depth
    print(f"Speculation depth (K): {speculation_depth}")

    results = []
    all_acceptance_lengths = []
    total_drafted = 0
    total_accepted = 0

    for i, prompt in enumerate(prompts):
        print(f"  Sample {i+1}/{len(prompts)}...", end=" ", flush=True)

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        original_length = input_ids.shape[1]
        generated = input_ids.clone()

        start = time.time()
        step_accepted = []

        # Generate with speculation
        for _ in range(max_tokens):
            if generated.shape[1] >= original_length + max_tokens:
                break

            # Generate draft hidden states from drafter's own predictions
            # P-EAGLE: Drafter predicts future hidden states from current context
            with torch.no_grad():
                drafter_outputs = drafter.forward(
                    input_ids=generated,
                    is_training=False
                )
                mtp_predictions = drafter_outputs["mtp_predictions"]

            # Convert hidden states to tokens
            draft_tokens = []
            for k in range(min(speculation_depth, max_tokens - (generated.shape[1] - original_length))):
                pred_hidden = mtp_predictions[k]
                logits = target_lm_head(pred_hidden)
                token_id = torch.argmax(logits, dim=-1)
                draft_tokens.append(token_id.item())

            # Verify with target
            accepted_count = 0
            draft_tensor = torch.tensor([draft_tokens], device="cuda")
            verification_input = torch.cat([generated, draft_tensor], dim=1)

            with torch.no_grad():
                verify_outputs = target_model(verification_input)
                verify_logits = verify_outputs.logits[0, generated.shape[1]-1:, :]

            for j, draft_token in enumerate(draft_tokens):
                target_token = torch.argmax(verify_logits[j]).item()
                if draft_token == target_token:
                    accepted_count += 1
                    total_accepted += 1
                else:
                    break

            # Append accepted tokens, or fall back to target's prediction
            if accepted_count > 0:
                new_tokens = torch.tensor([draft_tokens[:accepted_count]], device="cuda")
                generated = torch.cat([generated, new_tokens], dim=1)
            else:
                fallback_token = torch.argmax(verify_logits[0]).item()
                new_token = torch.tensor([[fallback_token]], device="cuda")
                generated = torch.cat([generated, new_token], dim=1)

            step_accepted.append(accepted_count)
            total_drafted += len(draft_tokens)

        elapsed = time.time() - start
        tokens_generated = generated.shape[1] - original_length
        mal = np.mean(step_accepted) if step_accepted else 0
        all_acceptance_lengths.extend(step_accepted)

        output_text = tokenizer.decode(generated[0][original_length:], skip_special_tokens=True)

        results.append({
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "output": output_text[:200] + "..." if len(output_text) > 200 else output_text,
            "tokens": tokens_generated,
            "accepted": sum(step_accepted),
            "mal": mal,
            "time": elapsed,
            "tps": tokens_generated / elapsed
        })

        print(f"{tokens_generated} tokens, MAL={mal:.2f}, {tokens_generated/elapsed:.1f} tps")

    # Calculate acceptance rates per head
    acceptance_by_head = calculate_head_acceptance(all_acceptance_lengths, speculation_depth)

    return {
        "drafter_checkpoint": drafter_checkpoint,
        "target_model": target_model_name,
        "speculation_depth": speculation_depth,
        "total_samples": len(prompts),
        "mean_mal": np.mean(all_acceptance_lengths) if all_acceptance_lengths else 0,
        "acceptance_rate": total_accepted / total_drafted * 100 if total_drafted > 0 else 0,
        "total_drafted": total_drafted,
        "total_accepted": total_accepted,
        "acceptance_by_head": acceptance_by_head,
        "samples": results
    }


def calculate_head_acceptance(acceptance_history: List[int], k: int) -> Dict[int, float]:
    """Calculate acceptance rate for each head position."""
    rates = {}
    for pos in range(1, k + 1):
        accepted_at_pos = sum(1 for x in acceptance_history if x >= pos)
        rates[pos] = accepted_at_pos / len(acceptance_history) if acceptance_history else 0.0
    return rates


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Evaluation")
    parser.add_argument("--drafter_checkpoint", default="./checkpoints/best_model")
    parser.add_argument("--target_model", default="google/gemma-3-4b-it")
    parser.add_argument("--test_prompts", default=None,
                        help="JSON file with test prompts")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--baseline", action="store_true",
                        help="Also evaluate raw target model")
    parser.add_argument("--domain_test", action="store_true", default=True)
    parser.add_argument("--output", default="evaluation_results.json")

    args = parser.parse_args()

    # Default test prompts
    if args.test_prompts:
        with open(args.test_prompts) as f:
            prompts = json.load(f)
    else:
        prompts = [
            "Explain how payment processing works in 3 steps.",
            "What are common causes of SDK integration failures?",
            "Summarize the key metrics for monitoring payment health.",
            "Describe the difference between synchronous and asynchronous callbacks.",
            "How would you troubleshoot a timeout error in a payment gateway?",
        ]

    results = {"config": vars(args)}

    # Evaluate P-EAGLE
    peagle_results = evaluate_peagle(
        args.drafter_checkpoint,
        args.target_model,
        prompts,
        args.max_tokens
    )
    results["peagle"] = peagle_results

    # Baseline comparison
    if args.baseline:
        raw_results = evaluate_raw_model(args.target_model, prompts, args.max_tokens)
        results["baseline"] = raw_results

        # Calculate speedup
        raw_tps = raw_results["mean_tps"]
        peagle_tps = np.mean([s["tps"] for s in peagle_results["samples"]])
        speedup = peagle_tps / raw_tps if raw_tps > 0 else 1.0

        print("\n" + "="*70)
        print("  SPEEDUP SUMMARY")
        print("="*70)
        print(f"  Raw Model TPS:    {raw_tps:.1f}")
        print(f"  P-EAGLE TPS:      {peagle_tps:.1f}")
        print(f"  Speedup:          {speedup:.2f}x")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("  EVALUATION COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output}")
    print(f"\nKey Metrics:")
    print(f"  Mean Acceptance Length (MAL): {peagle_results['mean_mal']:.2f}")
    print(f"  Acceptance by Head:")
    for pos, rate in sorted(peagle_results['acceptance_by_head'].items()):
        print(f"    Position {pos}: {rate*100:.1f}%")


if __name__ == "__main__":
    main()
