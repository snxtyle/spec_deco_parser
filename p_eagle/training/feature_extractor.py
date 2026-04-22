#!/usr/bin/env python3
"""
P-EAGLE Feature Extraction Script

Extracts tri-layer fused hidden states from Target Model for training the Drafter.
"""

import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
from pathlib import Path

from ..utils.feature_utils import EagleDataset, align_segments_to_tokens, fuse_tri_layer_features


class TriLayerConfig:
    """Configuration for which layers to extract from."""

    def __init__(self, model, mode: str = "early,middle,final"):
        self.num_layers = model.config.num_hidden_layers
        self.mode = mode

        if mode == "early,middle,final":
            self.layer_indices = [
                self.num_layers // 4,
                self.num_layers // 2,
                3 * self.num_layers // 4
            ]
        elif mode == "first,middle,last":
            self.layer_indices = [0, self.num_layers // 2, self.num_layers - 1]
        elif mode == "all":
            self.layer_indices = list(range(self.num_layers))
        else:
            self.layer_indices = [int(x) for x in mode.split(",")]

        print(f"Tri-Layer Config: extracting from layers {self.layer_indices}")


class FeatureExtractor:
    """Main feature extraction pipeline for P-EAGLE."""

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        quantization: str = "4bit",
        layer_config: str = "early,middle,final",
        fusion_mode: str = "mean",
        max_length: int = 2048,
        batch_size: int = 1,
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fusion_mode = fusion_mode
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device

        # Setup quantization
        self.quant_config = self._setup_quantization(quantization)

        # Load model and tokenizer
        print(f"Loading target model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.quant_config if quantization != "none" else None,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            output_hidden_states=True
        )
        self.model.eval()

        self.layer_config = TriLayerConfig(self.model, layer_config)
        print(f"Model loaded. Hidden dim: {self.model.config.hidden_size}")

    def _setup_quantization(self, mode: str):
        if mode == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    @torch.no_grad()
    def extract_sample(self, batch):
        try:
            conversation_text = batch["conversation_text"][0]
            messages = batch["original_messages"][0]
            segments = batch["segments"][0]

            inputs = self.tokenizer(
                conversation_text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

            all_hidden_states = outputs.hidden_states
            fused_hidden = fuse_tri_layer_features(
                all_hidden_states,
                self.layer_config.layer_indices,
                self.fusion_mode
            )

            token_level_mask = align_segments_to_tokens(
                messages,
                segments,
                self.tokenizer,
                input_ids[0]
            )

            return {
                "input_ids": input_ids[0].cpu(),
                "fused_hidden_states": fused_hidden[0].cpu(),
                "loss_mask": token_level_mask.cpu(),
                "attention_mask": attention_mask[0].cpu()
            }

        except Exception as e:
            print(f"Error extracting sample: {e}")
            return None

    def process_file(self, input_path: str, shard_size: int = 1000):
        dataset = EagleDataset(input_path, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_features = []
        shard_idx = 0

        for batch in tqdm(dataloader, desc="Extracting features"):
            features = self.extract_sample(batch)
            if features is not None:
                all_features.append(features)

            if len(all_features) >= shard_size:
                self._save_shard(all_features, input_path, shard_idx)
                all_features = []
                shard_idx += 1

        if all_features:
            self._save_shard(all_features, input_path, shard_idx)

        print(f"Extraction complete. Saved {shard_idx + 1} shards to {self.output_dir}")

    def _save_shard(self, features, input_path, shard_idx):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        hidden_states = torch.nn.utils.rnn.pad_sequence(
            [f["fused_hidden_states"] for f in features],
            batch_first=True,
            padding_value=0.0
        )
        loss_masks = torch.nn.utils.rnn.pad_sequence(
            [f["loss_mask"] for f in features],
            batch_first=True,
            padding_value=0
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0
        )

        input_name = Path(input_path).stem
        output_file = self.output_dir / f"{input_name}_shard{shard_idx:04d}.pt"

        torch.save({
            "input_ids": input_ids,
            "fused_hidden_states": hidden_states,
            "loss_mask": loss_masks,
            "attention_mask": attention_masks,
            "model_name": self.model_name,
            "layer_indices": self.layer_config.layer_indices,
            "fusion_mode": self.fusion_mode,
            "num_samples": len(features)
        }, output_file)

        print(f"Saved {output_file} ({len(features)} samples)")


def main():
    parser = argparse.ArgumentParser(description="P-EAGLE Feature Extraction")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_data", required=True)
    parser.add_argument("--output_dir", default="./features")
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--layers", default="early,middle,final")
    parser.add_argument("--fusion", default="mean", choices=["mean", "weighted", "concat"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=1000)

    args = parser.parse_args()

    extractor = FeatureExtractor(
        model_name=args.model_path,
        output_dir=args.output_dir,
        quantization=args.quantization,
        layer_config=args.layers,
        fusion_mode=args.fusion,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    extractor.process_file(args.input_data, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
