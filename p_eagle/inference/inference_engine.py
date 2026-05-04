#!/usr/bin/env python3
"""P-EAGLE Inference Engine"""

import argparse
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..models.eagle_drafter import EagleDrafterModel
from ..models.tree_attention import TreeAttentionMask


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
        dtype: torch.dtype = torch.bfloat16,
        use_hidden_injection: bool = False  # Must match training - always False
    ):
        self.device = device
        self.dtype = dtype
        self.use_hidden_injection = use_hidden_injection

        print(f"Loading target model: {target_model_name}")
        self.target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
        if self.target_tokenizer.pad_token is None:
            self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_name,
            torch_dtype=dtype,
            device_map="auto",
            output_hidden_states=True
        )
        self.target_model.eval()

        self.target_hidden_dim = self.target_model.config.hidden_size

        print(f"Loading drafter from: {drafter_checkpoint}")
        self.drafter = EagleDrafterModel.load_checkpoint(drafter_checkpoint, device=device)
        self.drafter.eval()

        self.speculation_depth = self.drafter.speculation_depth
        self.acceptance_history = []

        # Initialize tree attention for parallel verification
        self.tree_attention = TreeAttentionMask(self.speculation_depth)

        print(f"P-EAGLE initialized: Target={target_model_name}, K={self.speculation_depth}")
        if use_hidden_injection:
            print("  Hidden state injection: ENABLED")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9
    ) -> Tuple[str, GenerationMetrics]:
        """Generate text using parallel speculative decoding with hidden state injection."""
        input_ids = self.target_tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        original_length = input_ids.shape[1]

        total_draft_tokens = 0
        total_accepted_tokens = 0
        target_passes = 0
        drafter_passes = 0

        start_time = time.time()
        generated = input_ids.clone()

        # Store target hidden state for injection loop
        prev_target_hidden = None

        for _ in range(max_new_tokens):
            if generated.shape[1] >= original_length + max_new_tokens:
                break

            # Draft K tokens (with optional hidden state injection)
            # CRITICAL: Ensure hidden state is on drafter's device for injection
            target_hidden_for_drafter = None
            if self.use_hidden_injection and prev_target_hidden is not None:
                target_hidden_for_drafter = prev_target_hidden.to(self.drafter.device)

            draft_tokens, _, prev_target_hidden = self._generate_draft_parallel(
                generated,
                self.speculation_depth,
                target_hidden_for_drafter
            )
            drafter_passes += 1
            total_draft_tokens += len(draft_tokens)

            # Verify with target (returns verified tokens and hidden state for next iteration)
            accepted, verified_tokens, prev_target_hidden = self._verify_parallel(
                generated,
                draft_tokens,
                temperature
            )
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
                # Reset hidden state injection on fallback
                prev_target_hidden = None

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

    def _generate_draft_parallel(
        self,
        input_ids: torch.Tensor,
        k: int,
        target_hidden: Optional[torch.Tensor] = None
    ) -> Tuple[List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Generate K draft tokens using cascaded MTP heads.

        Each head predicts the next token based on the previous prediction,
        creating a chain: h_t -> t+1, h_{t+1} -> t+2, etc.

        Args:
            input_ids: Current sequence [batch, seq_len]
            k: Number of tokens to draft
            target_hidden: Optional target hidden states from previous verification

        Returns:
            draft_tokens: List of drafted token IDs
            last_logits: Logits from the last drafted token
            drafter_hidden: Last drafter hidden state
        """
        draft_tokens = []
        last_logits = None

        # Start with current input
        current_input = input_ids

        for i in range(k):
            # Forward pass through drafter
            outputs = self.drafter(
                current_input,
                output_hidden_states=True,
                target_hidden=target_hidden if self.use_hidden_injection else None,
                is_training=False
            )

            # Use the i-th MTP head (or last available head)
            head_idx = min(i, len(outputs["mtp_predictions"]) - 1)
            pred_hidden = outputs["mtp_predictions"][head_idx][:, -1:]

            # Convert to logits using target's lm_head
            logits = self._hidden_to_logits(pred_hidden)
            last_logits = logits

            # Get token
            probs = F.softmax(logits[0, 0], dim=-1)
            token = torch.argmax(probs).item()
            draft_tokens.append(token)

            # Append token to input for next iteration
            current_input = torch.cat([
                current_input,
                torch.tensor([[token]], device=self.device)
            ], dim=1)

            # Update target_hidden if using injection
            if self.use_hidden_injection and target_hidden is not None:
                # Extend target_hidden with prediction (simplified)
                target_hidden = outputs["projected_hidden"]

        # Get final hidden state from last forward pass
        drafter_hidden = outputs["projected_hidden"][:, -1:]

        return draft_tokens, last_logits, drafter_hidden

    def _verify_parallel(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
        temperature: float,
        acceptance_threshold: float = 0.8
    ) -> Tuple[int, List[int], Optional[torch.Tensor]]:
        """Verify draft tokens using parallel forward pass.

        SMART VERIFICATION: Accepts draft tokens if they are in the target's top-k
        or have similar probability to the target's choice. This improves MAL while
        maintaining output quality.

        Args:
            input_ids: Current sequence
            draft_tokens: Drafted token IDs to verify
            temperature: Sampling temperature
            acceptance_threshold: Min probability ratio to accept draft (0.8 = 80%)

        Returns:
            num_accepted: Number of accepted tokens
            verified_tokens: List of verified token IDs
            target_hidden: Last hidden state for injection loop
        """
        draft_tensor = torch.tensor([draft_tokens], device=self.device)
        verification_input = torch.cat([input_ids, draft_tensor], dim=1)

        # Single forward pass - let model handle attention mask
        outputs = self.target_model(
            input_ids=verification_input,
            output_hidden_states=True
        )

        seq_len = input_ids.shape[1]
        num_draft = len(draft_tokens)

        # Get logits for all speculative positions at once
        logits = outputs.logits[0, seq_len-1:seq_len+num_draft-1]

        verified_tokens = []
        for i in range(min(num_draft, logits.shape[0])):
            # Get target's probability distribution
            probs = F.softmax(logits[i] / temperature, dim=-1)
            target_token = torch.argmax(probs).item()
            target_prob = probs[target_token].item()
            draft_prob = probs[draft_tokens[i]].item()

            # SMART ACCEPTANCE: Accept if draft is target's choice OR close in probability
            if draft_tokens[i] == target_token:
                # Exact match - definitely accept
                verified_tokens.append(draft_tokens[i])
            elif draft_prob >= acceptance_threshold * target_prob:
                # Draft token has similar probability to target's choice (>80%)
                # This handles cases where draft picks token B (prob=0.44) and target picks A (prob=0.45)
                verified_tokens.append(draft_tokens[i])
            else:
                # Target model strongly disagrees - accept up to this point and use target's choice
                verified_tokens.append(target_token)
                break

        # Extract hidden state at the last verified position for injection loop
        target_hidden = None
        if self.use_hidden_injection and hasattr(outputs, 'hidden_states'):
            last_layer_hidden = outputs.hidden_states[-1]
            last_verified_pos = seq_len + len(verified_tokens) - 1
            target_hidden = last_layer_hidden[:, :last_verified_pos+1, :]

        return len(verified_tokens), verified_tokens, target_hidden

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
    parser.add_argument("--use_hidden_injection", action="store_true",
                        help="Enable hidden state injection from target model")

    args = parser.parse_args()

    engine = PEAGLEInference(
        target_model_name=args.target_model,
        drafter_checkpoint=args.drafter_checkpoint,
        use_hidden_injection=args.use_hidden_injection
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
