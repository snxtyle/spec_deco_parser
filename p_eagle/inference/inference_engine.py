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
        use_hidden_injection: bool = False
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
            draft_tokens, _, prev_target_hidden = self._generate_draft_parallel(
                generated,
                self.speculation_depth,
                prev_target_hidden
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
        """Generate K draft tokens using parallel MTP heads in a SINGLE forward pass.

        Args:
            input_ids: Current sequence [batch, seq_len]
            k: Number of tokens to draft
            target_hidden: Optional target hidden states from previous verification [batch, seq_len, hidden_dim]

        Returns:
            draft_tokens: List of drafted token IDs
            last_logits: Logits from the last drafted token (for debugging)
            drafter_hidden: Last drafter hidden state (unused, for compatibility)
        """
        # Single forward pass on the drafter (inference mode - no trimming)
        outputs = self.drafter(
            input_ids,
            output_hidden_states=True,
            target_hidden=target_hidden if self.use_hidden_injection else None,
            is_training=False
        )

        draft_tokens = []
        last_logits = None

        # Extract predictions from each MTP head in parallel
        for i in range(min(k, len(outputs["mtp_predictions"]))):
            # Get the last position's predicted hidden state from the i-th MTP head
            pred_hidden = outputs["mtp_predictions"][i][:, -1:]

            # Convert predicted hidden state to logits using target's lm_head
            logits = self._hidden_to_logits(pred_hidden)
            last_logits = logits

            # Deterministic drafting using argmax for maximum speed
            # Matches greedy verification strategy for consistent output
            probs = F.softmax(logits[0, 0], dim=-1)
            token = torch.argmax(probs).item()
            draft_tokens.append(token)

        # Return drafter's projected hidden for potential future use
        drafter_hidden = outputs["projected_hidden"][:, -1:]

        return draft_tokens, last_logits, drafter_hidden

    def _verify_parallel(
        self,
        input_ids: torch.Tensor,
        draft_tokens: List[int],
        temperature: float
    ) -> Tuple[int, List[int], Optional[torch.Tensor]]:
        """Verify draft tokens using tree attention for parallel verification.

        Uses TreeAttentionMask to create tree-style attention inputs that allow
        the target model to verify all speculative tokens in a single forward pass.

        VERIFICATION SAFETY: Uses greedy verification - only accepts a draft token
        if it matches the target model's argmax prediction. This ensures output
        quality matches the target model exactly.

        Returns:
            num_accepted: Number of accepted tokens
            verified_tokens: List of verified token IDs
            target_hidden: Last hidden state for injection loop
        """
        draft_tensor = torch.tensor([draft_tokens], device=self.device)

        # Create tree attention inputs using TreeAttentionMask
        full_input_ids, attention_mask, position_ids = self.tree_attention.create_tree_inputs(
            input_ids, draft_tensor
        )

        # Single parallel forward pass with tree attention
        outputs = self.target_model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True
        )

        seq_len = input_ids.shape[1]
        num_draft = len(draft_tokens)

        # Get logits for all speculative positions at once
        logits = outputs.logits[0, seq_len-1:seq_len+num_draft-1]

        verified_tokens = []
        for i in range(min(num_draft, logits.shape[0])):
            # GREEDY VERIFICATION: Accept only if draft token matches target's top choice
            target_token = torch.argmax(logits[i]).item()

            if draft_tokens[i] == target_token:
                verified_tokens.append(draft_tokens[i])
            else:
                # Target model disagrees - accept up to this point and use target's choice
                verified_tokens.append(target_token)
                break

        # Extract hidden state at the last verified position for injection loop
        # Get the last layer's hidden state at the final position of the verified sequence
        target_hidden = None
        if self.use_hidden_injection and hasattr(outputs, 'hidden_states'):
            # hidden_states is a tuple of (num_layers,) tensors [batch, seq_len, hidden_dim]
            last_layer_hidden = outputs.hidden_states[-1]  # [batch, seq_len + num_draft, hidden_dim]
            # Take the hidden state at the last verified position
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
