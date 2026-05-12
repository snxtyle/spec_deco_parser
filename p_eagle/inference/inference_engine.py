#!/usr/bin/env python3
"""
P-EAGLE Inference Script

Correct architecture: Drafter predicts hidden states → Target's lm_head → Tokens
"""

import argparse
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from p_eagle.models.peagle_drafter import EagleDrafterModel


def run_inference(
    target_model_name: str,
    drafter_checkpoint: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    device: str = "cuda",
):
    """Run P-EAGLE inference with correct hidden state prediction."""

    print(f"Loading target model: {target_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(target_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Disable CUDA graphs to avoid dynamic shape issues with TorchInductor
    torch._inductor.config.triton.cudagraphs = False

    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    target_model.eval()

    # Get target's lm_head for converting hidden states to tokens
    if hasattr(target_model, 'lm_head'):
        target_lm_head_fallback = target_model.lm_head
    elif hasattr(target_model, 'model') and hasattr(target_model.model, 'lm_head'):
        target_lm_head_fallback = target_model.model.lm_head
    else:
        raise ValueError("Could not find lm_head in target model")

    print(f"Loading P-EAGLE drafter: {drafter_checkpoint}")
    drafter = EagleDrafterModel.load_checkpoint(drafter_checkpoint, device=device)
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

    # Try to load saved lm_head from checkpoint for vocab compatibility
    import json
    from pathlib import Path
    checkpoint_path = Path(drafter_checkpoint)
    config_path = checkpoint_path / "config.json"

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        saved_vocab_size = config.get("vocab_size", None)
        current_vocab_size = len(tokenizer)

        if saved_vocab_size and saved_vocab_size != current_vocab_size:
            print(f"WARNING: Vocab size mismatch!")
            print(f"  Saved (training): {saved_vocab_size}")
            print(f"  Current (target): {current_vocab_size}")
            print(f"  Using target model's lm_head with {current_vocab_size} vocab")
            print(f"  This may cause token ID mismatches!")

    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    original_length = input_ids.shape[1]

    print(f"\nPrompt: {prompt}")
    print(f"Prompt tokens: {original_length}")
    print("-" * 50)

    # Import tree attention
    from p_eagle.models.tree_attention import TreeAttentionMask, verify_drafts_parallel

    # Initialize tree attention
    tree_attn = TreeAttentionMask(speculation_depth)

    # Generation loop
    generated = input_ids.clone()
    total_draft_tokens = 0
    total_accepted = 0
    target_passes = 0

    # KV-cache for efficient inference (if model supports it)
    past_key_values = None
    use_kv_cache = hasattr(target_model.config, 'use_cache')

    start_time = time.time()

    for step in range(max_new_tokens):
        if generated.shape[1] >= original_length + max_new_tokens:
            break

        current_seq_len = generated.shape[1]

        # EAGLE: First get target model's hidden state at current position
        # Use KV-cache if available for efficiency
        with torch.no_grad():
            if use_kv_cache and past_key_values is not None:
                # Only process the last token with KV-cache
                target_outputs = target_model(
                    generated[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=True
                )
                past_key_values = target_outputs.past_key_values
            else:
                target_outputs = target_model(generated, output_hidden_states=True)
                if use_kv_cache:
                    past_key_values = target_outputs.past_key_values

            target_hidden = target_outputs.hidden_states[-1]  # [batch, seq_len, hidden_dim]

        # Generate K draft hidden states using drafter with EAGLE-3 hidden injection
        # CRITICAL: Pass target_hidden for concatenation at first layer
        with torch.no_grad():
            drafter_outputs = drafter.forward(
                input_ids=generated,
                target_hidden=target_hidden,  # EAGLE-3: Concatenation injection
                is_training=False
            )
            # mtp_predictions is a list of K tensors, each [batch, 1, target_hidden_dim]
            mtp_predictions = drafter_outputs["mtp_predictions"]

        # Convert predicted hidden states to tokens using target's lm_head
        draft_tokens = []
        for k in range(min(speculation_depth, max_new_tokens - (generated.shape[1] - original_length))):
            # Get predicted hidden for position k
            pred_hidden = mtp_predictions[k]  # [batch, 1, target_hidden_dim]

            # Pass through target's lm_head to get logits
            logits = target_lm_head(pred_hidden)  # [batch, 1, vocab_size]

            # Apply temperature sampling
            if temperature > 0 and temperature != 1.0:
                logits = logits / temperature

            if temperature == 0:
                # Greedy decode
                token_id = torch.argmax(logits, dim=-1)  # [batch, 1]
            else:
                # Sample from distribution
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs[0], num_samples=1).unsqueeze(0)

            draft_tokens.append(token_id.item())

        total_draft_tokens += len(draft_tokens)

        # Verify with target model using tree attention for parallel verification
        draft_tensor = torch.tensor([draft_tokens], device=device)

        # Use tree attention for efficient parallel verification
        with torch.no_grad():
            tree_inputs = tree_attn.prepare_tree_inputs(generated, draft_tensor)

            # Single forward pass with tree attention mask
            verify_outputs = target_model(
                input_ids=tree_inputs["input_ids"],
                attention_mask=tree_inputs["attention_mask"],
                position_ids=tree_inputs["position_ids"]
            )

            # Get logits for speculative positions
            verified_len = tree_inputs["verified_len"]
            verify_logits = verify_outputs.logits[0, verified_len-1:verified_len-1+len(draft_tokens), :]

        # Accept tokens greedily (sequential acceptance is still correct)
        accepted = []
        for i, draft_token in enumerate(draft_tokens):
            target_token = torch.argmax(verify_logits[i]).item()
            if draft_token == target_token:
                accepted.append(draft_token)
                total_accepted += 1
            else:
                # Accept target's choice and stop
                accepted.append(target_token)
                break

        if accepted:
            new_tokens = torch.tensor([accepted], device=device)
            generated = torch.cat([generated, new_tokens], dim=1)
            # Reset KV-cache when sequence changes
            past_key_values = None

        target_passes += 1

        # Print progress
        current_text = tokenizer.decode(
            generated[0][original_length:], skip_special_tokens=True
        )
        print(f"\rStep {step+1}: {len(accepted)} tokens accepted", end="")

    wall_time = time.time() - start_time
    print("\n" + "-" * 50)

    # Final output
    output_text = tokenizer.decode(
        generated[0][original_length:], skip_special_tokens=True
    )

    print(f"\nGenerated text:\n{output_text}")
    print("\n" + "=" * 50)
    print("Metrics:")
    print(f"  Total tokens generated: {generated.shape[1] - original_length}")
    print(f"  Draft tokens proposed: {total_draft_tokens}")
    print(f"  Tokens accepted: {total_accepted}")
    print(f"  Target forward passes: {target_passes}")
    print(f"  Mean Acceptance Length: {total_accepted/target_passes:.2f}")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Tokens/sec: {(generated.shape[1] - original_length) / wall_time:.2f}")


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Inference")
    parser.add_argument("--target_model", required=True, help="Target model name")
    parser.add_argument("--drafter_checkpoint", required=True, help="Path to P-EAGLE checkpoint")
    parser.add_argument("--prompt", default="Explain quantum computing in simple terms.")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    run_inference(
        target_model_name=args.target_model,
        drafter_checkpoint=args.drafter_checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=args.device,
    )


if __name__ == "__main__":
    main()
