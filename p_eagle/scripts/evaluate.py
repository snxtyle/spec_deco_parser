#!/usr/bin/env python3
"""
P-EAGLE Comprehensive Evaluation Script
Measures MAL, throughput, acceptance rates, and domain-specific performance.
"""

import argparse
import json
import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from ..inference.inference_engine import PEAGLEInference
from transformers import AutoTokenizer, AutoModelForCausalLM


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
    """Evaluate P-EAGLE enhanced model."""
    print("\n" + "="*70)
    print("  EVALUATING P-EAGLE MODEL")
    print("="*70)

    engine = PEAGLEInference(
        target_model_name=target_model_name,
        drafter_checkpoint=drafter_checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    results = []
    all_acceptance_lengths = []

    for i, prompt in enumerate(prompts):
        print(f"  Sample {i+1}/{len(prompts)}...", end=" ")

        output, metrics = engine.generate(prompt, max_new_tokens=max_tokens,
                                          temperature=0.7, top_p=0.9)

        all_acceptance_lengths.extend(engine.acceptance_history)

        results.append({
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "output": output[:200] + "..." if len(output) > 200 else output,
            "tokens": metrics.total_tokens,
            "accepted": metrics.accepted_tokens,
            "mal": metrics.mean_acceptance_length,
            "time": metrics.wall_time,
            "tps": metrics.total_tokens / metrics.wall_time,
            "speedup_vs_naive": metrics.speedup
        })

        print(f"{metrics.total_tokens} tokens, MAL={metrics.mean_acceptance_length:.2f}, "
              f"{metrics.total_tokens/metrics.wall_time:.1f} tps")

    # Calculate acceptance rates per head
    acceptance_by_head = calculate_head_acceptance(all_acceptance_lengths, engine.speculation_depth)

    return {
        "drafter_checkpoint": drafter_checkpoint,
        "target_model": target_model_name,
        "speculation_depth": engine.speculation_depth,
        "total_samples": len(prompts),
        "mean_mal": np.mean(all_acceptance_lengths) if all_acceptance_lengths else 0,
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


def domain_specific_test_juspay(engine: PEAGLEInference) -> Dict:
    """Domain-specific tests for Juspay logs."""
    print("\n" + "="*70)
    print("  DOMAIN TEST: JUSPAY LOGS")
    print("="*70)

    # Test 1: System header (should have near-perfect acceptance)
    system_header = """<agent-identity>
You are BitBot, a technical support specialist.
</agent-identity>

User: bitbot-api: analyze_sdk_logs"""

    _, metrics = engine.generate(system_header, max_new_tokens=50)
    header_mal = metrics.mean_acceptance_length

    print(f"\n1. System Header Test:")
    print(f"   MAL: {header_mal:.2f} (expected: high for memorized content)")

    # Test 2: Tool call structure prediction
    tool_prompt = """Given the logs, please"""

    output, metrics = engine.generate(tool_prompt, max_new_tokens=30)
    tool_mal = metrics.mean_acceptance_length

    print(f"\n2. Tool Call Initiation Test:")
    print(f"   MAL: {tool_mal:.2f}")
    print(f"   Output preview: {output[:100]}...")

    return {
        "system_header_mal": header_mal,
        "tool_call_mal": tool_mal
    }


def lossless_verification(engine: PEAGLEInference, target_model_name: str,
                          prompts: List[str]) -> Dict:
    """Verify P-EAGLE produces identical output to raw model (deterministic)."""
    print("\n" + "="*70)
    print("  LOSSLESS VERIFICATION CHECK")
    print("="*70)

    # Note: True losslessness requires greedy decoding (temp=0)
    matches = 0
    total = len(prompts)

    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    target = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    target.eval()

    for i, prompt in enumerate(prompts):
        # Greedy generation from target
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(target.device)
        with torch.no_grad():
            target_out = target.generate(input_ids, max_new_tokens=20, do_sample=False)
        target_text = tokenizer.decode(target_out[0][input_ids.shape[1]:], skip_special_tokens=True)

        # P-EAGLE generation (greedy)
        peagle_text, _ = engine.generate(prompt, max_new_tokens=20, temperature=0.0)

        match = target_text.strip() == peagle_text.strip()
        matches += int(match)

        status = "PASS" if match else "DIFF"
        print(f"  Sample {i+1}: {status}")

    return {
        "verified": matches,
        "total": total,
        "match_rate": matches / total
    }


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Evaluation")
    parser.add_argument("--drafter_checkpoint", default="./checkpoints/best_model")
    parser.add_argument("--target_model", default="google/gemma-7b")
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

    # Domain tests
    if args.domain_test:
        engine = PEAGLEInference(
            target_model_name=args.target_model,
            drafter_checkpoint=args.drafter_checkpoint
        )
        domain_results = domain_specific_test_juspay(engine)
        results["domain_tests"] = domain_results

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
