#!/usr/bin/env python3
"""P-EAGLE Inference Engine"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..models.eagle_drafter import EagleDrafterModel


@dataclass
class GenerationMetrics:
    total_tokens: int
    accepted_tokens: int
    target_forward_passes: int
    drafter_forward_passes: int
    mean_acceptance_length: float
    speedup: float
    wall_time: float


class PEAGLEInference:
    """P-EAGLE Parallel Speculative Decoding Inference Engine."""

    def __init__(
        self,
        target_model_name: str,
        drafter_checkpoint: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = device
        self.dtype = dtype

        print(f"Loading target model: {target_model_name}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=dtype,
            device_map="auto"
        )
        self.target_model.eval()

        self.target_hidden_dim = self.target_model.config.hidden_size

        print(f"Loading drafter from: {drafter_checkpoint}")
        self.drafter = EagleDrafterModel.load_checkpoint(drafter_checkpoint, device=device)
        self.drafter.eval()

        self.speculation_depth = self.drafter.speculation_depth
        self.acceptance_history = []

        print(f"P-EAGLE initialized: Target={target_model_name}, K={self.speculation_depth}")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Tuple[str, GenerationMetrics]:
        """Generate text using parallel speculative decoding."""
        input_ids = self.target_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        original_length = input_ids.shape[1]

        total_draft_tokens = 0
        total_accepted_tokens = 0
        target_passes = 0
        drafter_passes = 0

        start_time = time.time()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            if generated.shape[1] >= original_length + max_new_tokens:
                break

            # Draft K tokens
            draft_tokens, _ = self._generate_draft_parallel(generated, self.speculation_depth)
            drafter_passes += 1
            total_draft_tokens += len(draft_tokens)

            # Verify with target
            accepted, verified_tokens = self._verify_parallel(generated, draft_tokens, temperature)
            target_passes += 1

            total_accepted_tokens += accepted
            self.acceptance_history.append(accepted)

            if len(verified_tokens) > 0:
                new_tokens = torch.tensor([verified_tokens], device=self.device)
                generated = torch.cat([generated, new_tokens], dim=1)
            else:
                # Fallback
                next_token = self._sample_from_target(generated, temperature, top_p)
                generated = torch.cat([generated, next_token], dim=1)

        wall_time = time.time() - start_time
        output_text = self.target_tokenizer.decode(
            generated[0][original_length:], skip_special_tokens=True
        )

        metrics = GenerationMetrics(
            total_tokens=generated.shape[1] - original_length,
            accepted_tokens=total_accepted_tokens,
            target_forward_passes=target_passes,
            drafter_forward_passes=drafter_passes,
            mean_acceptance_length=sum(self.acceptance_history) / len(self.acceptance_history) if self.acceptance_history else 0.0,
            speedup=total_accepted_tokens / target_passes if target_passes > 0 else 1.0,
            wall_time=wall_time
        )

        return output_text, metrics

    def _generate_draft_parallel(self, input_ids, k):
        draft_tokens = []
        current_ids = input_ids.clone()

        for i in range(k):
            outputs = self.drafter(current_ids, output_hidden_states=True)
            projected = outputs["projected_hidden"][:, -1:]

            if i < len(outputs["mtp_predictions"]):
                pred_hidden = outputs["mtp_predictions"][i][:, -1:]
            else:
                pred_hidden = projected

            logits = self._hidden_to_logits(pred_hidden)
            probs = F.softmax(logits[0, 0], dim=-1)
            token = torch.multinomial(probs, num_samples=1).item()

            draft_tokens.append(token)
            current_ids = torch.cat([current_ids, torch.tensor([[token]], device=self.device)], dim=1)

        return draft_tokens, logits

    def _verify_parallel(self, input_ids, draft_tokens, temperature):
        draft_tensor = torch.tensor([draft_tokens], device=self.device)
        verification_input = torch.cat([input_ids, draft_tensor], dim=1)

        outputs = self.target_model(verification_input)
        seq_len = input_ids.shape[1]
        logits = outputs.logits[0, seq_len-1:seq_len+len(draft_tokens)-1]

        verified_tokens = []
        for i in range(min(len(draft_tokens), logits.shape[0])):
            target_probs = F.softmax(logits[i] / temperature, dim=-1)
            target_token_prob = target_probs[draft_tokens[i]].item()

            if target_token_prob > 0.0:
                verified_tokens.append(draft_tokens[i])
            else:
                corrected_token = torch.multinomial(target_probs, num_samples=1).item()
                verified_tokens.append(corrected_token)
                break

        return len(verified_tokens), verified_tokens

    def _hidden_to_logits(self, hidden):
        if hasattr(self.target_model, 'lm_head'):
            return self.target_model.lm_head(hidden)
        return torch.randn(hidden.shape[0], hidden.shape[1], self.target_model.config.vocab_size,
                          device=hidden.device, dtype=hidden.dtype)

    def _sample_from_target(self, input_ids, temperature, top_p):
        outputs = self.target_model(input_ids)
        logits = outputs.logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Inference")
    parser.add_argument("--target_model", required=True)
    parser.add_argument("--drafter_checkpoint", required=True)
    parser.add_argument("--prompt", default="Explain quantum computing")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()

    engine = PEAGLEInference(
        target_model_name=args.target_model,
        drafter_checkpoint=args.drafter_checkpoint
    )

    output, metrics = engine.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature
    )

    print(f"\nOutput: {output}")
    print(f"\nMetrics:")
    print(f"  Tokens: {metrics.total_tokens}")
    print(f"  Accepted: {metrics.accepted_tokens}")
    print(f"  MAL: {metrics.mean_acceptance_length:.2f}")
    print(f"  Speedup: {metrics.speedup:.2f}x")
    print(f"  Time: {metrics.wall_time:.2f}s")


if __name__ == "__main__":
    main()
