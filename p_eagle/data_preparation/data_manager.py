"""
DataManager - A modular data loading class for training datasets.

Supports:
- HuggingFace and local data sources
- Streaming mode for memory efficiency
- OpenAI format output (for inference)
- Training format (input/output pairs for training)
- Batch retrieval with state maintenance
"""

import json
import math
import os
import random
import re
import hashlib
import shutil
import subprocess
import tempfile
import zipfile
import urllib.request
from typing import Dict, Any, Callable, Optional, Iterator, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Try importing datasets library
try:
    from datasets import load_dataset, IterableDataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not installed. Run: pip install datasets")

# Try importing dotenv for .env file support
try:
    from dotenv import load_dotenv
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", os.getenv("HF_TOKEN"))
except ImportError:
    HF_TOKEN = None


# ============================================
# Data Classes
# ============================================

@dataclass
class Sample:
    """Represents a single training sample."""
    # Original format (OpenAI messages)
    messages: List[Dict[str, Any]]

    # Training format (input/output pairs)
    input_text: str = ""
    output_text: str = ""
    input_messages: List[Dict[str, Any]] = field(default_factory=list)
    output_messages: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    source: str = ""
    dataset_name: str = ""
    index: int = 0

    def to_openai_format(self) -> Dict[str, Any]:
        """Return data in OpenAI messages format."""
        return {"messages": self.messages}

    def to_training_format(self) -> Dict[str, Any]:
        """Return data in training format (input/output)."""
        return {
            "input": self.input_text,
            "output": self.output_text,
            "input_messages": self.input_messages,
            "output_messages": self.output_messages
        }


@dataclass
class Batch:
    """Represents a batch of samples."""
    samples: List[Sample]
    start_index: int
    end_index: int
    total_available: int

    def __len__(self):
        return len(self.samples)

    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Return batch in OpenAI format."""
        return [s.to_openai_format() for s in self.samples]

    def to_training_format(self) -> List[Dict[str, Any]]:
        """Return batch in training format."""
        return [s.to_training_format() for s in self.samples]


# ============================================
# DataManager Class
# ============================================

class DataManager:
    """
    A modular data loading class supporting multiple sources and formats.

    Usage:
        dm = DataManager(source="hf", dataset_name="Salesforce/xlam-function-calling-60k")
        dm = DataManager(source="local", data_files="./data/*.json")

        # Streaming iteration
        for sample in dm:
            print(sample)

        # Batch retrieval with state
        batch = dm.get_batch(10)  # Get first 10
        batch = dm.get_batch(10)  # Get next 10
    """

    def __init__(
        self,
        source: str,
        dataset_name: Optional[str] = None,
        split: str = "train",
        data_files: Optional[str] = None,
        streaming: bool = True,
        transform_fn: Optional[Callable[[Dict], Dict]] = None,
        shuffle: bool = False,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None
    ):
        """
        Initialize DataManager.

        Args:
            source: Data source - "hf" for HuggingFace, "local" for local files
            dataset_name: HuggingFace dataset name (required for hf source)
            split: Dataset split to load (default: "train")
            data_files: Path to local files (required for local source)
            streaming: Use streaming mode (memory efficient, default: True)
            transform_fn: Custom transform function to convert samples
            shuffle: Shuffle the data (only works in non-streaming mode)
            seed: Random seed for shuffling
            cache_dir: Cache directory for HuggingFace datasets
            token: HuggingFace token for gated datasets
        """
        if not HF_AVAILABLE and source == "hf":
            raise ImportError("datasets library required for HuggingFace source. Install: pip install datasets")

        self.source = source
        self.dataset_name = dataset_name
        self.split = split
        self.data_files = data_files
        self.streaming = streaming
        self.shuffle = shuffle
        self.seed = seed
        self.cache_dir = cache_dir
        self.token = token or HF_TOKEN

        # Transform function
        self.transform_fn = transform_fn or self.default_transform

        # State for batch retrieval
        self._cursor = 0
        self._total_samples = None
        self._is_exhausted = False

        # Load the dataset
        self.dataset = self._load_dataset()

        # Apply shuffle if not streaming (streaming doesn't support shuffle)
        if self.shuffle and not self.streaming:
            if hasattr(self.dataset, 'shuffle'):
                self.dataset = self.dataset.shuffle(seed=self.seed)

    def _load_dataset(self):
        """Load dataset based on source."""
        if self.source == "hf":
            if not self.dataset_name:
                raise ValueError("dataset_name required for HuggingFace source")

            return load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=self.streaming,
                cache_dir=self.cache_dir,
                token=self.token if self.token else False
            )

        elif self.source == "local":
            if not self.data_files:
                raise ValueError("data_files required for local source")

            # For local files, load manually to handle complex JSON structures
            from glob import glob
            import os

            files = glob(self.data_files)
            if not files:
                raise ValueError(f"No files found matching: {self.data_files}")

            # Load all JSON files manually
            all_data = []
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Handle array format - check if it's a list of messages (conversations)
                    if isinstance(data, list):
                        # Check if it's a list of message objects (one conversation per file)
                        if len(data) > 0 and isinstance(data[0], dict) and "role" in data[0]:
                            # This is a list of messages = ONE conversation
                            # Wrap it as a single sample
                            all_data.append({"messages": data})
                        else:
                            # It's a list of samples
                            all_data.extend(data)
                    elif isinstance(data, dict):
                        all_data.append(data)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
                    continue

            if not all_data:
                raise ValueError("No valid data loaded from local files")

            # Return as iterable
            if self.streaming:
                return iter(all_data)
            else:
                # Return as list-like for non-streaming
                return all_data

        else:
            raise ValueError(f"Invalid source: {source}. Use 'hf' or 'local'")

    def _normalize_content(self, content: Any) -> str:
        """
        Normalize content field that can be:
        - Simple string
        - List of {"type": "text", "text": "..."} objects
        - Stringified JSON (e.g., "[{'role':...}, ...]")
        - Nested dict like {'role': ..., 'content': ...}
        - None
        """
        if content is None:
            return ""

        if isinstance(content, str):
            # Check if it's a stringified JSON - try to parse and handle properly
            if content.startswith("[") or content.startswith("{"):
                try:
                    parsed = json.loads(content)
                    return self._normalize_content(parsed)
                except:
                    pass
            return content

        if isinstance(content, list):
            # Handle list of {"type": "text", "text": "..."} objects
            if len(content) > 0 and isinstance(content[0], dict):
                # Check if it's a list of messages
                if "role" in content[0] and "content" in content[0]:
                    # This is a list of messages - extract as conversation string
                    return self._extract_content_from_messages(content)
                # Otherwise it's text blocks
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            texts.append(item["text"])
                        elif "value" in item:
                            texts.append(item["value"])
                return "\n".join(texts)
            return str(content)

        if isinstance(content, dict):
            # Check if it's a nested message dict like {'role': ..., 'content': ...}
            if "role" in content and "content" in content:
                # This is a nested message - extract role and content
                role = content.get("role", "unknown")
                inner_content = content.get("content", "")
                # Recursively normalize the inner content
                inner_str = self._normalize_content(inner_content)
                return f"{role}: {inner_str}"
            # Otherwise it's just a dict, convert to string
            return str(content)

        return str(content)

    def _extract_content_from_messages(self, messages: List[Dict]) -> str:
        """Extract and combine content from a list of messages."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Handle list content
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                if text_parts:
                    parts.append(f"{role}: {' '.join(text_parts)}")
        return "\n".join(parts)

    def _extract_nested_messages(self, content: str) -> List[Dict[str, Any]]:
        """Extract nested messages from a stringified conversation."""
        import ast

        if not isinstance(content, str):
            return []

        # Try to parse as Python literal (since data uses single quotes)
        try:
            parsed = ast.literal_eval(content)
        except:
            # Try JSON as fallback
            try:
                parsed = json.loads(content)
            except:
                return []

        # If parsed is a list of messages, extract them
        if isinstance(parsed, list):
            messages = []
            for msg in parsed:
                if isinstance(msg, dict) and "role" in msg:
                    # Recursively normalize content
                    normalized_content = self._normalize_content(msg.get("content"))
                    messages.append({
                        "role": msg.get("role"),
                        "content": normalized_content
                    })
            return messages

        # If parsed is a dict with messages key
        if isinstance(parsed, dict) and "messages" in parsed:
            messages = []
            for msg in parsed.get("messages", []):
                if isinstance(msg, dict) and "role" in msg:
                    normalized_content = self._normalize_content(msg.get("content"))
                    messages.append({
                        "role": msg.get("role"),
                        "content": normalized_content
                    })
            return messages

        return []

    def default_transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default transform function - converts various formats to OpenAI messages format.

        Supports:
        - OpenAI format: {"messages": [...]}
        - Direct list format: [{"role": ..., "content": ...}, ...]
        - Nested conversation format: {"role": "user", "content": "[{'role': ..., 'content': ...}]"}
        - ShareGPT format: {"conversations": [...]}
        - Instruction-output format: {"instruction": ..., "output": ...}
        - xlam format: {"query": ..., "answers": ..., "tools": ...}
        """
        # Already a list of messages (direct format from local files)
        if isinstance(example, list):
            normalized_messages = []
            for msg in example:
                if isinstance(msg, dict) and "role" in msg:
                    normalized_msg = dict(msg)
                    normalized_msg["content"] = self._normalize_content(msg.get("content"))
                    normalized_messages.append(normalized_msg)
            if normalized_messages:
                return {"messages": normalized_messages}

        # Already in OpenAI format with "messages" key
        if "messages" in example:
            # Normalize content in messages
            messages = example["messages"]
            normalized_messages = []
            for msg in messages:
                content = msg.get("content", "")

                # Check if content contains nested messages
                nested = self._extract_nested_messages(content)
                if nested:
                    # Add nested messages directly instead of the wrapper
                    for nested_msg in nested:
                        normalized_messages.append(nested_msg)
                else:
                    # Normalize and add the original message
                    normalized_msg = dict(msg)
                    normalized_msg["content"] = self._normalize_content(content)
                    normalized_messages.append(normalized_msg)
            return {"messages": normalized_messages}

        # ShareGPT format
        if "conversations" in example:
            return {
                "messages": [
                    {"role": "user" if turn["from"] == "human" else "assistant", "content": turn["value"]}
                    for turn in example["conversations"]
                ]
            }

        # Instruction-output format
        if "instruction" in example and "output" in example:
            return {
                "messages": [
                    {"role": "user", "content": example["instruction"]},
                    {"role": "assistant", "content": example["output"]}
                ]
            }

        # xlam format (from Salesforce/xlam-function-calling-60k)
        if "query" in example and "answers" in example:
            query = example.get("query", "")
            answers = example.get("answers", "")

            # Parse tool calls from answers
            tool_calls_data = self._parse_tool_calls(answers)

            messages = [{"role": "user", "content": query}]

            if tool_calls_data:
                assistant_tool_calls = []
                for idx, tc in enumerate(tool_calls_data):
                    name = tc.get("name", "")
                    arguments = tc.get("arguments", {})

                    if isinstance(arguments, dict):
                        arguments = json.dumps(arguments)

                    assistant_tool_calls.append({
                        "id": f"call_{idx}_{name}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments
                        }
                    })

                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls
                })

            return {"messages": messages}

        # Fallback: convert entire example to user message
        return {"messages": [{"role": "user", "content": str(example)}]}

    def _parse_tool_calls(self, answers_str: str) -> List[Dict[str, Any]]:
        """Parse tool calls from answers field."""
        import ast

        # Try JSON first
        try:
            tool_calls = json.loads(answers_str)
            if isinstance(tool_calls, list):
                return tool_calls
        except:
            pass

        # Try literal_eval
        try:
            tool_calls = ast.literal_eval(answers_str)
            if isinstance(tool_calls, list):
                return tool_calls
        except:
            pass

        return []

    def _convert_to_training_format(self, messages: List[Dict[str, Any]]) -> Tuple[str, str, List[Dict], List[Dict]]:
        """
        Convert OpenAI messages to training format.

        Training format:
        - input: user message(s) + tool definitions (if any)
        - output: assistant message(s) + tool calls (if any)

        Returns:
            Tuple of (input_text, output_text, input_messages, output_messages)
        """
        user_messages = []
        assistant_messages = []
        tool_definitions = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")

            if role == "system":
                # System messages go to input
                user_messages.append({"role": "system", "content": content})

            elif role == "user":
                user_messages.append({"role": "user", "content": content})

            elif role == "assistant":
                # Check for tool calls
                if tool_calls:
                    # Tool calls go to output
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        name = func.get("name", "")
                        args = func.get("arguments", "")

                        if isinstance(args, dict):
                            args = json.dumps(args)

                        assistant_messages.append({
                            "role": "assistant",
                            "tool_calls": [{
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": args
                                }
                            }]
                        })

                if content:
                    assistant_messages.append({"role": "assistant", "content": content})

            elif role == "tool":
                # Tool results - include in input
                user_messages.append({"role": "tool", "content": content})

        # Build input text
        input_parts = []
        for msg in user_messages:
            if msg.get("content"):
                input_parts.append(f"{msg['role']}: {msg['content']}")
        input_text = "\n".join(input_parts)

        # Build output text
        output_parts = []
        for msg in assistant_messages:
            if msg.get("content"):
                output_parts.append(f"assistant: {msg['content']}")
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    output_parts.append(f"[TOOL_CALL] {func.get('name')}({func.get('arguments')})")
        output_text = "\n".join(output_parts)

        return input_text, output_text, user_messages, assistant_messages

    def _process_sample(self, raw_sample: Dict[str, Any], index: int) -> Optional[Sample]:
        """Process a raw sample into a Sample object."""
        try:
            # Apply transform
            transformed = self.transform_fn(raw_sample)
            messages = transformed.get("messages", [])

            if not messages:
                return None

            # Normalize all message content to strings
            normalized_messages = []
            for msg in messages:
                normalized_msg = dict(msg)
                normalized_msg["content"] = self._normalize_content(msg.get("content"))
                normalized_messages.append(normalized_msg)
            messages = normalized_messages

            # Convert to training format
            input_text, output_text, input_msgs, output_msgs = self._convert_to_training_format(messages)

            return Sample(
                messages=messages,
                input_text=input_text,
                output_text=output_text,
                input_messages=input_msgs,
                output_messages=output_msgs,
                index=index
            )

        except Exception as e:
            return None

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over samples (streaming mode)."""
        # Handle both iterator and list formats for local data
        if isinstance(self.dataset, list):
            # Non-streaming mode - iterate over list
            for idx, raw_sample in enumerate(self.dataset):
                sample = self._process_sample(raw_sample, idx)
                if sample:
                    yield sample
        else:
            # Streaming mode - iterate over iterator
            for idx, raw_sample in enumerate(self.dataset):
                sample = self._process_sample(raw_sample, idx)
                if sample:
                    yield sample

    def __len__(self) -> int:
        """Get total number of samples (only works in non-streaming mode)."""
        if self.streaming:
            raise ValueError("Cannot get length in streaming mode. Set streaming=False first.")

        if self._total_samples is None:
            self._total_samples = len(self.dataset)

        return self._total_samples

    def get_batch(self, batch_size: int, reset: bool = False) -> Batch:
        """
        Get a batch of samples.

        Args:
            batch_size: Number of samples to retrieve
            reset: If True, reset cursor to beginning

        Returns:
            Batch object containing samples
        """
        if reset:
            self._cursor = 0
            self._is_exhausted = False

        samples = []
        start_index = self._cursor

        if self.streaming:
            # Streaming mode: iterate from cursor
            count = 0
            for raw_sample in self.dataset:
                if count < self._cursor:
                    count += 1
                    continue

                sample = self._process_sample(raw_sample, count)
                if sample:
                    samples.append(sample)
                    self._cursor += 1

                if len(samples) >= batch_size:
                    break

                count += 1

            if not samples:
                self._is_exhausted = True

        else:
            # Non-streaming mode: use index-based access
            if self._total_samples is None:
                self._total_samples = len(self.dataset)

            idx = self._cursor
            count = 0

            while idx < self._total_samples and count < batch_size:
                try:
                    raw_sample = self.dataset[idx]
                    sample = self._process_sample(raw_sample, idx)

                    if sample:
                        samples.append(sample)
                        count += 1

                    self._cursor = idx + 1

                except Exception:
                    pass

                idx += 1

            if idx >= self._total_samples:
                self._is_exhausted = True

        end_index = self._cursor

        return Batch(
            samples=samples,
            start_index=start_index,
            end_index=end_index,
            total_available=self._total_samples or -1
        )

    @property
    def is_exhausted(self) -> bool:
        """Check if all samples have been consumed."""
        return self._is_exhausted

    def reset(self):
        """Reset the cursor to beginning."""
        self._cursor = 0
        self._is_exhausted = False

    def save_to_jsonl(
        self,
        output_path: str,
        limit: Optional[int] = None,
        format: str = "openai"
    ):
        """
        Save samples to JSONL file.

        Args:
            output_path: Path to output file
            limit: Maximum number of samples to save
            format: "openai" or "training"
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        count = 0
        with open(output_path, "w") as f:
            for sample in self:
                if format == "openai":
                    data = sample.to_openai_format()
                else:
                    data = sample.to_training_format()

                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1

                if limit and count >= limit:
                    break

        print(f"Saved {count} samples to {output_path}")


# ============================================
# Factory Functions
# ============================================

def create_data_manager(
    source: str = "hf",
    dataset_name: Optional[str] = None,
    data_files: Optional[str] = None,
    **kwargs
) -> DataManager:
    """
    Factory function to create a DataManager instance.

    Example:
        # HuggingFace
        dm = create_data_manager("hf", "Salesforce/xlam-function-calling-60k")

        # Local files
        dm = create_data_manager("local", data_files="./data/*.json")
    """
    return DataManager(
        source=source,
        dataset_name=dataset_name,
        data_files=data_files,
        **kwargs
    )


# ============================================
# Main (for testing)
# ============================================
# ============================================
# CLI Interface (matches generate_data.py style)
# ============================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="DataManager - Load and process datasets for training"
    )

    # Output configuration
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "-f", "--format",
        type=str,
        choices=["openai", "sharegpt"],
        default="openai",
        help="Output format: openai or sharegpt (default: openai)"
    )

    # Number of samples
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=100,
        help="Total number of data points to process"
    )

    # Source selection
    parser.add_argument(
        "--hf",
        action="store_true",
        help="Use HuggingFace dataset"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local files"
    )

    # Source options
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./processed",
        help="Input directory for local files (default: ./processed)"
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="Salesforce/xlam-function-calling-60k",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load (default: train)"
    )

    # Ratio for mixing sources
    parser.add_argument(
        "--ratio",
        type=str,
        default=None,
        help="Ratio for mixing sources (e.g., hf:0.6,local:0.4). Required when using hf+local."
    )

    # Batch configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for get_batch (default: 10)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Number of batches to retrieve. If not set, gets all samples at once."
    )

    # Other options
    parser.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="Minimum words in assistant response (0 = no filter)"
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=False,
        help="Remove duplicate samples based on content hash"
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output file format: jsonl or json (default: jsonl)"
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Shuffle the data (only works in non-streaming mode)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        default=False,
        help="Disable streaming mode (loads all data into memory)"
    )

    # Resume capability
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing output file"
    )

    args = parser.parse_args()

    # Determine streaming mode
    streaming = not args.no_streaming

    # Determine sources
    sources = []
    if args.hf:
        sources.append("hf")
    if args.local:
        sources.append("local")

    if not sources:
        # Default to local if no source specified
        sources = ["local"]

    # Handle single source vs multiple sources
    if len(sources) == 1:
        source = sources[0]

        if source == "hf":
            dm = DataManager(
                source="hf",
                dataset_name=args.hf_dataset,
                split=args.split,
                streaming=streaming,
                shuffle=args.shuffle,
                seed=args.seed
            )
        elif source == "local":
            dm = DataManager(
                source="local",
                data_files=f"{args.input_dir}/*.json",
                streaming=streaming,
                shuffle=args.shuffle,
                seed=args.seed
            )

        # Get samples
        if args.num_batches:
            # Batch mode
            print(f"\n{'='*60}")
            print(f"RETRIEVING BATCHES")
            print(f"{'='*60}")
            print(f"Batch size: {args.batch_size}")
            print(f"Number of batches: {args.num_batches}")

            all_samples = []
            for i in range(args.num_batches):
                batch = dm.get_batch(args.batch_size)
                print(f"\nBatch {i+1}: {len(batch.samples)} samples (index {batch.start_index} - {batch.end_index})")

                if batch.samples:
                    sample = batch.samples[0]
                    print(f"  First sample input: {sample.input_text[:80]}...")
                    print(f"  First sample output: {sample.output_text[:80]}...")

                all_samples.extend(batch.samples)

                if dm.is_exhausted:
                    print("  (Dataset exhausted)")
                    break

            final_samples = all_samples[:args.num_samples]
        else:
            # All samples mode
            print(f"\n{'='*60}")
            print(f"LOADING DATA")
            print(f"{'='*60}")
            print(f"Source: {source}")
            print(f"Streaming: {streaming}")

            final_samples = []
            for i, sample in enumerate(dm):
                if i >= args.num_samples:
                    break
                final_samples.append(sample)

                if (i + 1) % 100 == 0:
                    print(f"  Loaded {i + 1} samples...")

            print(f"  Loaded {len(final_samples)} samples")

    else:
        # Multiple sources - handle ratio
        print(f"\n{'='*60}")
        print(f"MULTI-SOURCE MODE")
        print(f"{'='*60}")

        # Parse ratio
        ratio = {}
        if args.ratio:
            for part in args.ratio.split(","):
                part = part.strip()
                if ":" in part:
                    key, value = part.split(":", 1)
                    ratio[key.strip()] = float(value.strip())
        else:
            # Default equal ratio
            ratio = {src: 1.0 / len(sources) for src in sources}

        print(f"Sources: {', '.join(sources)}")
        print(f"Ratio: {args.ratio or 'equal'}")

        # Load from each source
        all_samples = []
        for src in sources:
            if src == "hf":
                dm = DataManager(
                    source="hf",
                    dataset_name=args.hf_dataset,
                    split=args.split,
                    streaming=streaming,
                    shuffle=args.shuffle,
                    seed=args.seed
                )
            elif src == "local":
                dm = DataManager(
                    source="local",
                    data_files=f"{args.input_dir}/*.json",
                    streaming=streaming,
                    shuffle=args.shuffle,
                    seed=args.seed
                )

            target = int(args.num_samples * ratio.get(src, 0))
            print(f"\nLoading {target} samples from {src}...")

            count = 0
            for sample in dm:
                if count >= target:
                    break
                all_samples.append(sample)
                count += 1

            print(f"  Loaded {count} samples")

        # Shuffle if requested
        if args.shuffle:
            import random
            random.shuffle(all_samples)

        final_samples = all_samples[:args.num_samples]

    print(f"\n{'='*60}")
    print(f"PROCESSING")
    print(f"{'='*60}")
    print(f"Total samples: {len(final_samples)}")

    # Deduplicate if requested
    if args.deduplicate:
        print("\nDeduplicating...")
        seen_hashes = set()
        unique_samples = []
        for sample in final_samples:
            h = hashlib.sha256(
                (sample.input_text + sample.output_text).encode()
            ).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_samples.append(sample)
        print(f"  Removed {len(final_samples) - len(unique_samples)} duplicates")
        final_samples = unique_samples

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output, exist_ok=True)

    if args.output_format == "json":
        output_file = os.path.join(args.output, f"dataset_{timestamp}.json")
    else:
        output_file = os.path.join(args.output, f"dataset_{timestamp}.jsonl")

    # Route through EAGLEDistiller to generate loss_mask_segments and apply persona handling
    print(f"\nProcessing through EAGLE distillation pipeline...")

    # Convert samples to LiteLLM format for EAGLEDistiller
    litellm_samples = []
    for idx, sample in enumerate(final_samples):
        litellm_format = {
            "type": "SUCCESS",
            "conversation": {"messages": sample.messages},
            "response": {},
            "_source": sources[idx % len(sources)] if len(sources) > 1 else sources[0]
        }
        litellm_samples.append(litellm_format)

    # Run through EAGLEDistiller for proper mask generation and persona handling
    distiller = EAGLEDistiller(
        output_dir=args.output,
        target_samples=len(litellm_samples),
        enable_deduplication=args.deduplicate
    )

    # Process samples with source-aware handling
    processed_samples = []
    for raw_data in litellm_samples:
        source = raw_data.pop("_source", "local")
        processed = distiller._process_sample(raw_data, source=source)
        if processed:
            processed_samples.append(processed)

    # Write unified output with loss_mask_segments
    output_file = os.path.join(args.output, f"dataset_{timestamp}.jsonl")
    with open(output_file, "w") as f:
        for sample in processed_samples:
            f.write(json.dumps({
                "messages": sample["messages"],
                "loss_mask_segments": sample["loss_mask_segments"]
            }, ensure_ascii=False) + "\n")

    print(f"Written {len(processed_samples)} samples with loss_mask_segments")

    # Also print sample info
    if final_samples:
        print(f"\n{'='*60}")
        print(f"SAMPLE PREVIEW")
        print(f"{'='*60}")
        sample = final_samples[0]

        if args.format == "openai" or args.format == "sharegpt":
            print("\nMessages:")
            for msg in sample.messages[:3]:
                content = str(msg.get("content", ""))[:50]
                print(f"  {msg.get('role')}: {content}...")
        else:
            print(f"\nInput: {sample.input_text[:100]}...")
            print(f"Output: {sample.output_text[:100]}...")


# ============================================
# Smart Secret Scanning Module
# ============================================
# Uses multiple detection strategies:
# 1. detect-secrets (Yelp) - ML-enhanced pattern detection
# 2. trufflehog (Truffle Security) - Deep git scanning + entropy
# 3. Presidio (Microsoft) - PII detection
# 4. Entropy analysis - Statistical anomaly detection
# 5. Custom ML heuristics - Beyond regex

@dataclass
class SecretFinding:
    """Represents a detected secret in content."""
    file_path: str
    line_number: int
    secret_type: str
    severity: str  # HIGH, MEDIUM, LOW
    confidence: float  # 0.0 to 1.0
    masked_value: str
    context: str
    detection_method: str  # Which scanner found it


class EntropyAnalyzer:
    """Statistical analysis to find high-entropy (random) strings that may be secrets."""

    @staticmethod
    def calculate_entropy(string: str) -> float:
        """Calculate Shannon entropy of a string. Higher = more random = likely secret."""
        if len(string) < 8:
            return 0.0

        prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy

    @staticmethod
    def is_likely_secret(token: str) -> Tuple[bool, float]:
        """
        Determine if a token is likely a secret based on multiple heuristics.
        Returns (is_secret, confidence_score)
        """
        if len(token) < 20:
            return False, 0.0

        # Skip obvious non-secrets
        if token.isalpha() or token.isdigit():
            return False, 0.0

        # Calculate entropy
        entropy = EntropyAnalyzer.calculate_entropy(token)

        # Secret characteristics
        has_upper = any(c.isupper() for c in token)
        has_lower = any(c.islower() for c in token)
        has_digit = any(c.isdigit() for c in token)
        has_special = any(not c.isalnum() for c in token)
        char_types = sum([has_upper, has_lower, has_digit, has_special])

        # High-entropy thresholds for base64/hex/random strings
        score = 0.0

        # Entropy scoring (normalized for typical token lengths)
        if entropy > 5.5:  # Very high entropy
            score += 0.4
        elif entropy > 4.5:  # High entropy
            score += 0.25

        # Character diversity scoring
        if char_types >= 3:
            score += 0.3
        elif char_types >= 2:
            score += 0.15

        # Length bonus (longer = more likely to be a generated secret)
        if len(token) > 40:
            score += 0.2
        elif len(token) > 30:
            score += 0.1

        # Pattern bonuses
        if re.match(r'^[A-Za-z0-9_\-]+$', token):  # Alphanumeric + underscore/hyphen
            score += 0.1

        # Penalties
        if any(common in token.lower() for common in ['example', 'test', 'sample', 'demo', 'fake', 'mock']):
            score -= 0.5

        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', token, re.I):
            score -= 0.3  # UUIDs are usually not secrets

        return score > 0.5, min(max(score, 0.0), 1.0)


class SmartSecretScanner:
    """
    Intelligent secret scanner using multiple detection engines.
    No hardcoded regex - relies on statistical analysis and external libraries.
    """

    def __init__(self, stop_on_secret: bool = False, entropy_threshold: float = 0.7):
        self.stop_on_secret = stop_on_secret
        self.entropy_threshold = entropy_threshold
        self.all_findings: List[SecretFinding] = []

        # Check available scanners
        self._has_detect_secrets = self._check_detect_secrets()
        self._has_trufflehog = self._check_trufflehog()
        self._has_presidio = self._check_presidio()

        self.entropy_analyzer = EntropyAnalyzer()

    def _check_detect_secrets(self) -> bool:
        """Check if Yelp's detect-secrets is available."""
        try:
            subprocess.run(["detect-secrets", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _check_trufflehog(self) -> bool:
        """Check if trufflehog is available."""
        try:
            subprocess.run(["trufflehog", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _check_presidio(self) -> bool:
        """Check if Microsoft Presidio is available."""
        try:
            import presidio_analyzer
            return True
        except ImportError:
            return False

    def _scan_with_detect_secrets(self, content: str, source_hint: str) -> List[SecretFinding]:
        """Use Yelp's detect-secrets library."""
        findings = []

        if not self._has_detect_secrets:
            return findings

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_path = f.name

            result = subprocess.run(
                ["detect-secrets", "scan", temp_path, "--all-files", "--force-use-all-plugins"],
                capture_output=True,
                text=True,
                timeout=30
            )

            os.unlink(temp_path)

            if result.stdout:
                try:
                    scan_results = json.loads(result.stdout)
                    for file_path, secrets in scan_results.get("results", {}).items():
                        for secret in secrets:
                            findings.append(SecretFinding(
                                file_path=source_hint,
                                line_number=secret.get("line_number", 1),
                                secret_type=secret.get("type", "Unknown Secret"),
                                severity="HIGH",
                                confidence=secret.get("confidence", 0.8),
                                masked_value=f"[{secret.get('type', 'Secret')}]",
                                context=content.split('\n')[secret.get("line_number", 1) - 1][:200],
                                detection_method="detect-secrets"
                            ))
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            pass

        return findings

    def _scan_with_trufflehog(self, content: str, source_hint: str) -> List[SecretFinding]:
        """Use TruffleHog for deep scanning."""
        findings = []

        if not self._has_trufflehog:
            return findings

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                temp_path = f.name

            result = subprocess.run(
                ["trufflehog", "filesystem", temp_path, "--json", "--no-update"],
                capture_output=True,
                text=True,
                timeout=30
            )

            os.unlink(temp_path)

            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                try:
                    finding_data = json.loads(line)
                    detector = finding_data.get("DetectorName", "Unknown")
                    raw = finding_data.get("Raw", "")

                    # Skip test/demo data
                    if any(test in raw.lower() for test in ['example', 'test', 'fake', 'sample']):
                        continue

                    findings.append(SecretFinding(
                        file_path=source_hint,
                        line_number=finding_data.get("SourceMetadata", {}).get("Data", {}).get("Filesystem", {}).get("line", 1),
                        secret_type=detector,
                        severity="HIGH",
                        confidence=0.9,
                        masked_value=f"[{detector}]",
                        context=raw[:200],
                        detection_method="trufflehog"
                    ))
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            pass

        return findings

    def _scan_with_presidio(self, content: str, source_hint: str) -> List[SecretFinding]:
        """Use Microsoft Presidio for PII detection."""
        findings = []

        if not self._has_presidio:
            return findings

        try:
            from presidio_analyzer import AnalyzerEngine

            analyzer = AnalyzerEngine()
            results = analyzer.analyze(text=content, language='en')

            for result in results:
                # Only high-confidence PII
                if result.score < 0.7:
                    continue

                # Map entity types to severity
                high_risk_pii = ['CREDIT_CARD', 'CRYPTO', 'IBAN_CODE', 'US_PASSPORT', 'US_SSN']
                severity = "HIGH" if result.entity_type in high_risk_pii else "MEDIUM"

                # Get context
                start = max(0, result.start - 50)
                end = min(len(content), result.end + 50)
                context = content[start:end]

                findings.append(SecretFinding(
                    file_path=source_hint,
                    line_number=content[:result.start].count('\n') + 1,
                    secret_type=result.entity_type,
                    severity=severity,
                    confidence=result.score,
                    masked_value=f"[{result.entity_type}]",
                    context=context,
                    detection_method="presidio"
                ))

        except Exception as e:
            pass

        return findings

    def _scan_with_entropy(self, content: str, source_hint: str) -> List[SecretFinding]:
        """Use statistical entropy analysis to find potential secrets."""
        findings = []

        # Tokenize content looking for candidate strings
        # Look for: base64 strings, hex strings, alphanumeric sequences
        candidates = re.findall(r'[A-Za-z0-9+/=_-]{20,}', content)

        for candidate in candidates:
            is_secret, confidence = self.entropy_analyzer.is_likely_secret(candidate)

            if is_secret and confidence >= self.entropy_threshold:
                # Find line number
                line_num = content[:content.find(candidate)].count('\n') + 1

                findings.append(SecretFinding(
                    file_path=source_hint,
                    line_number=line_num,
                    secret_type="High-Entropy String (Potential Secret)",
                    severity="MEDIUM",
                    confidence=confidence,
                    masked_value="[High-Entropy String]",
                    context=candidate[:200],
                    detection_method="entropy-analysis"
                ))

        return findings

    def scan_sample(self, content: str, sample_id: str = "") -> Tuple[bool, List[SecretFinding]]:
        """
        Scan content using all available scanners.

        Returns:
            (is_clean, findings)
        """
        all_findings = []

        # Run all scanners
        scanners = [
            ("detect-secrets", self._scan_with_detect_secrets),
            ("trufflehog", self._scan_with_trufflehog),
            ("presidio", self._scan_with_presidio),
            ("entropy", self._scan_with_entropy),
        ]

        for scanner_name, scanner_func in scanners:
            try:
                findings = scanner_func(content, sample_id)
                all_findings.extend(findings)
            except Exception as e:
                pass  # Continue with other scanners

        # Deduplicate findings by context/location
        seen = set()
        unique_findings = []
        for f in all_findings:
            key = (f.line_number, f.context[:50])
            if key not in seen:
                seen.add(key)
                unique_findings.append(f)

        if unique_findings:
            self.all_findings.extend(unique_findings)
            return False, unique_findings

        return True, []

    def get_masked_content(self, content: str, findings: List[SecretFinding]) -> str:
        """Redact detected secrets from content."""
        masked = content
        lines = content.split('\n')

        for finding in findings:
            line_idx = finding.line_number - 1
            if 0 <= line_idx < len(lines):
                line = lines[line_idx]
                # Simple redaction: replace the specific substring
                # In production, use spaCy/NER for precise span detection
                if finding.context in line:
                    lines[line_idx] = line.replace(finding.context, finding.masked_value)
                elif len(finding.context) > 10:
                    # Try to find and mask similar content
                    for word in line.split():
                        if len(word) > 15 and similar(word, finding.context[:len(word)]) > 0.8:
                            lines[line_idx] = lines[line_idx].replace(word, finding.masked_value)

        return '\n'.join(lines)

    def mask_messages(self, messages: List[Dict[str, Any]], findings: List[SecretFinding]) -> List[Dict[str, Any]]:
        """Mask secrets in message content."""
        if not findings:
            return messages

        # Convert messages to string, mask, then parse back
        content = json.dumps(messages, indent=2)
        masked_content = self.get_masked_content(content, findings)

        try:
            # Try to parse back
            masked_data = json.loads(masked_content)
            if isinstance(masked_data, list):
                return masked_data
        except json.JSONDecodeError:
            pass

        # Fallback: mask individual message fields
        masked_messages = []
        for msg in messages:
            masked_msg = dict(msg)
            for field in ['content', 'text', 'value']:
                if field in masked_msg and isinstance(masked_msg[field], str):
                    text = masked_msg[field]
                    for finding in findings:
                        if finding.context in text:
                            text = text.replace(finding.context, finding.masked_value)
                    masked_msg[field] = text
            masked_messages.append(masked_msg)

        return masked_messages

    def print_summary(self):
        """Print detection summary."""
        if not self.all_findings:
            print("\n✓ No secrets detected")
            return

        print("\n" + "="*60)
        print("SECRET SCAN SUMMARY")
        print("="*60)

        by_method = {}
        by_severity = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        for f in self.all_findings:
            by_method.setdefault(f.detection_method, 0)
            by_method[f.detection_method] += 1
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

        print(f"\nTotal findings: {len(self.all_findings)}")
        print(f"  - HIGH severity: {by_severity['HIGH']}")
        print(f"  - MEDIUM severity: {by_severity['MEDIUM']}")
        print(f"  - LOW severity: {by_severity['LOW']}")

        print("\nDetection methods:")
        for method, count in sorted(by_method.items(), key=lambda x: -x[1]):
            print(f"  - {method}: {count}")

        if self.stop_on_secret:
            print("\n⚠️  Processing stopped early due to secret detection!")


def similar(a: str, b: str) -> float:
    """Calculate string similarity using difflib."""
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()


class SecretScanner:
    """Integrated secret scanner for EAGLE pipeline."""

    def __init__(self, stop_on_secret: bool = False):
        self.stop_on_secret = stop_on_secret
        self.all_findings: List[SecretFinding] = []
        self.has_gitleaks = find_gitleaks_cmd() is not None

    def scan_sample(self, messages: List[Dict[str, Any]], sample_id: str = "") -> Tuple[bool, List[SecretFinding]]:
        """
        Scan a conversation sample for secrets.

        Returns:
            (is_clean, findings)
        """
        # Convert messages to string for scanning
        content = json.dumps(messages, indent=2)

        # Try gitleaks first, fallback to regex
        if self.has_gitleaks:
            findings = scan_with_gitleaks(content, source_hint=sample_id)
        else:
            findings = scan_content_for_secrets(content, source_hint=sample_id)

        if findings:
            self.all_findings.extend(findings)
            return False, findings

        return True, []

    def get_masked_messages(self, messages: List[Dict[str, Any]], findings: List[SecretFinding]) -> List[Dict[str, Any]]:
        """Return messages with secrets masked."""
        masked_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if content:
                # Mask secrets in content
                for finding in findings:
                    # Extract the actual secret value from context and mask it
                    content = self._mask_in_text(content, finding)
            masked_msg = {**msg, "content": content}
            masked_messages.append(masked_msg)
        return masked_messages

    def _mask_in_text(self, text: str, finding: SecretFinding) -> str:
        """Apply masking for a specific finding."""
        # High-entropy detection heuristic
        words = text.split()
        masked_words = []
        for word in words:
            # Check if word looks like a secret (long, high entropy)
            if len(word) > 15 and self._is_high_entropy(word) and finding.secret_type.lower() in word.lower():
                masked_words.append(finding.masked_value)
            elif len(word) > 20 and self._is_high_entropy(word):
                masked_words.append(finding.masked_value)
            else:
                masked_words.append(word)
        return ' '.join(masked_words)

    def _is_high_entropy(self, s: str) -> bool:
        """Check if string has high character entropy (indicates randomness)."""
        if len(s) < 8:
            return False
        unique_chars = len(set(s))
        return unique_chars / len(s) > 0.7

    def print_summary(self):
        """Print secret scanning summary."""
        if not self.all_findings:
            return

        print("\n" + "="*60)
        print("SECRET SCAN SUMMARY")
        print("="*60)

        high = [f for f in self.all_findings if f.severity == "HIGH"]
        medium = [f for f in self.all_findings if f.severity == "MEDIUM"]

        print(f"Secrets found: {len(self.all_findings)}")
        print(f"  - HIGH severity: {len(high)}")
        print(f"  - MEDIUM severity: {len(medium)}")

        if self.stop_on_secret:
            print("\n⚠️  Processing stopped early due to secret detection!")


# ============================================
# EAGLE Speculative Decoding Data Distillation
# ============================================
# This module transforms raw LiteLLM logs into "Golden Dataset" for EAGLE training.
# It performs zero-cost (no LLM calls) cleaning using metadata and structural heuristics.
#
# Pipeline Overview:
# 1. Metadata-First Filtering: Check response codes and error indicators
# 2. Secret Scanning: Detect and mask sensitive data before model exposure
# 3. Adaptive Persona Trimming: Dynamically detect and strip filler phrases from system prompt
# 4. Structural Heuristics: Validate tool calls, code blocks, and response length
# 5. Loss Masking: Generate token-level masks for EAGLE training
# 6. Filter-and-Refill: Auto-replenish batches to meet target count
# 7. Semantic Deduplication: Remove duplicate queries using content hashing

class EAGLEDistiller:
    """
    Transforms raw LiteLLM conversation logs into training-ready data for EAGLE speculative decoding.

    This class implements a zero-cost data cleaning pipeline that uses metadata inspection
    and structural heuristics to filter and prepare high-quality training samples.

    Usage:
        distiller = EAGLEDistiller(input_dir="./raw", output_dir="./golden")
        distiller.run(target_samples=1000, batch_size=1000)

    Output:
        - golden_dataset.jsonl: Cleaned conversations in OpenAI format
        - loss_masks.jsonl: Token-level loss masks (1=train, 0=ignore)
    """

    # Technical keywords that indicate a coding/query task requiring code blocks
    TECHNICAL_KEYWORDS = [
        "haskell", "pr", "diff", "code", "function", "class", "def ",
        "import ", "return ", "async", "await", "rust", "python",
        "javascript", "typescript", "java", "go", "sql", "bash",
        "shell", "script", "compile", "debug", "error", "exception",
        "api", "endpoint", "json", "xml", "yaml", "config"
    ]

    # Error indicators to scan in content (more specific patterns)
    ERROR_INDICATORS = [
        "error_code", "ERROR:", "Bad Request", "MISSING_MANDATORY_PARAMETER",
        "authentication_error", "rate_limit_exceeded", "timeout_error",
        "401 Unauthorized", "403 Forbidden", "404 Not Found",
        "500 Internal Server Error", "invalid_request_error",
        "ServiceUnavailableError", "RateLimitError"
    ]

    def __init__(
        self,
        input_dir: str = "./raw",
        output_dir: str = "./golden",
        target_samples: int = 1000,
        batch_size: int = 1000,
        min_response_length: int = 30,
        enable_deduplication: bool = True,
        enable_code_validation: bool = False,
        enable_secret_scanning: bool = True,
        stop_on_secret: bool = False,
        mask_secrets: bool = True
    ):
        """
        Initialize the EAGLE Distiller.

        Args:
            input_dir: Directory containing raw LiteLLM JSON log files
            output_dir: Directory to save cleaned golden dataset
            target_samples: Target number of clean samples to produce
            batch_size: Number of raw samples to process per batch
            min_response_length: Minimum character length for assistant responses
            enable_deduplication: Whether to remove duplicate user queries
            enable_code_validation: Whether to require code blocks for code requests
            enable_secret_scanning: Whether to scan for secrets (API keys, tokens, passwords)
            stop_on_secret: Whether to stop processing when secrets are found
            mask_secrets: Whether to mask detected secrets in output
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_samples = target_samples
        self.batch_size = batch_size
        self.min_response_length = min_response_length
        self.enable_deduplication = enable_deduplication
        self.enable_code_validation = enable_code_validation
        self.enable_secret_scanning = enable_secret_scanning
        self.stop_on_secret = stop_on_secret
        self.mask_secrets = mask_secrets

        # Cache for dynamic regex pattern
        self._filler_pattern: Optional[re.Pattern] = None

        # Secret scanner
        self._secret_scanner: Optional[SmartSecretScanner] = None
        if enable_secret_scanning:
            self._secret_scanner = SmartSecretScanner(stop_on_secret=stop_on_secret)

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "filtered_metadata": 0,
            "filtered_error_content": 0,
            "filtered_no_code_blocks": 0,
            "filtered_short_response": 0,
            "filtered_tool_error": 0,
            "filtered_duplicate": 0,
            "filtered_secrets": 0,
            "masked_secrets": 0,
            "final_count": 0
        }

    def _load_raw_files(self) -> List[Dict[str, Any]]:
        """Load all raw JSON files from input directory."""
        import glob as glob_module

        files = glob_module.glob(os.path.join(self.input_dir, "*.json"))
        all_samples = []

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_samples.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                continue

        return all_samples

    def _extract_messages(self, raw_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract messages from LiteLLM log format.

        Handles the conversion from LiteLLM's internal format to OpenAI message format.
        LiteLLM stores conversation under 'conversation.messages' and response under 'response'.
        """
        conversation = raw_data.get("conversation", {})
        messages = conversation.get("messages", [])

        # Convert to OpenAI format
        converted = []
        for msg in messages:
            converted.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        # Add assistant response as the last message
        response = raw_data.get("response", {})
        if response.get("content"):
            converted.append({
                "role": "assistant",
                "content": response.get("content", "")
            })

        # Handle tool calls - if assistant made tool calls, add tool results
        if response.get("tool_calls"):
            # Find tool role messages that may have executed results
            # In LiteLLM logs, tool results might be in additional messages
            pass  # Tool results would need to be in the conversation

        return converted

    def _check_metadata_filter(self, raw_data: Dict[str, Any], verbose: bool = False) -> bool:
        """
        Metadata-First Filtering: Check response status.

        Discards entries where:
        - type field is not "SUCCESS"
        - Any response code indicates failure

        Returns True if sample passes, False if should be filtered.
        """
        # Check top-level type field
        response_type = raw_data.get("type", "")
        if response_type != "SUCCESS":
            self.stats["filtered_metadata"] += 1
            if verbose:
                print(f"    [FILTER] Metadata: type='{response_type}' (expected SUCCESS)")
            return False

        return True

    def _check_error_content(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Scan content for error indicators.

        Discards samples containing error keys in user/assistant messages only.
        Excludes tool messages (system outputs) from error checking.
        """
        # Only check user and assistant messages (not tool results)
        relevant_messages = [
            m for m in messages
            if m.get("role") in ("user", "assistant", "system")
        ]

        content_text = " ".join([
            str(m.get("content", "") or "") for m in relevant_messages
        ]).lower()

        for error in self.ERROR_INDICATORS:
            if error.lower() in content_text:
                self.stats["filtered_error_content"] += 1
                return False

        return True

    def _extract_filler_phrases(self, system_content: Any) -> List[str]:
        """
        Adaptive Persona Trimming: Extract filler phrases from system prompt.

        Dynamically parses the JAR system prompt to find the section listing
        filler phrases (e.g., "Certainly!", "Of course!", "Sure!").

        Args:
            system_content: The system prompt content (can be string or list)

        Returns:
            List of filler phrases to remove from assistant responses
        """
        # Default filler phrases if not found in system prompt
        default_fillers = [
            "Certainly!", "Of course!", "Sure!", "Absolutely!",
            "I'd be happy to", "I'd love to", "Here's", "Here's the"
        ]

        # Extract string content (handle list format)
        content_str = self._extract_content_string(system_content)

        # Look for a section in the system prompt that lists filler words/phrases
        # Common patterns: "filler phrases:", "avoid:", "don't say:", "phrases like:"
        filler_section_pattern = re.compile(
            r"(?:filler|avoid|preamble|intro)(?:\s+phrases?|\s+words?|\s+like)?\s*[:\-]\s*([^\n]+)",
            re.IGNORECASE
        )

        match = filler_section_pattern.search(content_str)
        if match:
            # Extract phrases from the matched section
            section_text = match.group(1)
            # Split by common delimiters
            phrases = re.split(r"[,;|]+", section_text)
            # Clean up and filter
            fillers = [p.strip().strip('"').strip("'") for p in phrases if p.strip()]
            if fillers:
                return fillers

        return default_fillers

    def _build_filler_pattern(self, messages: List[Dict[str, Any]]) -> re.Pattern:
        """
        Build or retrieve cached regex pattern for filler phrase removal.

        Uses the first system prompt found to extract filler phrases dynamically.
        """
        if self._filler_pattern is not None:
            return self._filler_pattern

        # Find system message
        system_content = None
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content")
                break

        # Extract filler phrases from system prompt
        fillers = self._extract_filler_phrases(system_content)

        # Build regex pattern - match filler at start of assistant response
        # Escape special regex characters in fillers
        escaped_fillers = [re.escape(f) for f in fillers]
        pattern_str = r"^(" + "|".join(escaped_fillers) + r")\s*"

        self._filler_pattern = re.compile(pattern_str, re.IGNORECASE)
        return self._filler_pattern

    def _trim_filler_phrases(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove filler phrases from beginning of assistant responses.

        Dynamically detects filler phrases from system prompt and strips them.
        """
        if not messages:
            return messages

        # Build the filler pattern
        pattern = self._build_filler_pattern(messages)

        # Process messages
        cleaned = []
        for msg in messages:
            if msg.get("role") == "assistant":
                raw_content = msg.get("content")
                content = self._extract_content_string(raw_content)
                if content:
                    # Strip filler phrases from beginning
                    content = pattern.sub("", content)
                    msg = {**msg, "content": content}
            cleaned.append(msg)

        return cleaned

    def _check_tool_integrity(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Structural Heuristic: Verify tool-call integrity.

        If assistant message contains tool_calls, ensure the following
        tool role message is not an error response.
        """
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Check if there's a following tool message
                if i + 1 < len(messages):
                    next_msg = messages[i + 1]
                    if next_msg.get("role") == "tool":
                        tool_content = str(next_msg.get("content", "")).lower()
                        # Check if tool result contains error indicators
                        for error in self.ERROR_INDICATORS[:5]:  # Use main errors
                            if error.lower() in tool_content:
                                self.stats["filtered_tool_error"] += 1
                                return False

        return True

    def _extract_content_string(self, content: Any) -> str:
        """
        Extract string content from message, handling both string and list formats.

        LiteLLM can return content as:
        - Simple string
        - List of content blocks (for multimodal)
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Extract text from list of content blocks
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
                    elif "text" in block:
                        parts.append(str(block["text"]))
            return " ".join(parts)
        return str(content)

    def _check_code_blocks(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Structural Heuristic: Code block validation.

        Only requires code blocks when user explicitly requests code/logs/diffs.
        Can be disabled via enable_code_validation flag.
        """
        # Skip if code validation is disabled
        if not self.enable_code_validation:
            return True

        # Find user message
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                raw_content = msg.get("content", "")
                user_content = self._extract_content_string(raw_content).lower()
                break

        # Only check if user explicitly asks for code/implementation
        code_request_patterns = [
            "write code", "show me", "implement", "create a function",
            "generate code", "create a class", "how do i write",
            "create the", "write the", "implement a", "build a"
        ]

        requests_code = any(pattern in user_content for pattern in code_request_patterns)

        if requests_code:
            # Find assistant message
            assistant_content = ""
            for msg in messages:
                if msg.get("role") == "assistant":
                    raw_content = msg.get("content", "")
                    assistant_content = self._extract_content_string(raw_content)
                    break

            # Check for markdown code blocks
            if "```" not in assistant_content:
                self.stats["filtered_no_code_blocks"] += 1
                return False

        return True

    def _check_response_length(self, messages: List[Dict[str, Any]]) -> bool:
        """
        Structural Heuristic: Length constraint.

        Keep samples if there's at least one substantial assistant response
        (100+ chars) - provides enough chain-of-thought for EAGLE to learn.
        """
        # Check if there's at least one assistant message that meets length
        has_substantial = False

        for msg in messages:
            if msg.get("role") == "assistant":
                content = str(msg.get("content") or "")
                if len(content) >= self.min_response_length:
                    has_substantial = True
                    break

        if not has_substantial:
            self.stats["filtered_short_response"] += 1
            return False

        return True

    def _generate_loss_mask_segments(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        EAGLE Preparation: Generate SEGMENT-BASED loss mask specification.

        CRITICAL: EAGLE-3 trains on TOKENS, not characters. This method outputs
        MASK SEGMENTS that specify which MESSAGE ROLES to train on. The Feature
        Extraction script will convert these to token-level masks using the ACTUAL
        GLM 5.1 tokenizer on the H200 cluster.

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

    # Master system prompt for EAGLE training when no system message exists
    MASTER_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer user questions accurately and concisely."""

    def _inject_jar_persona(self, messages: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """
        Source-Aware Persona Management with System Message Enforcement.

        CRITICAL: EAGLE training requires system message to be FIRST in conversation.
        This ensures consistent hidden state patterns for the draft model.

        Rules:
        - If first message is system: keep it (local) or replace with master (HF)
        - If no system message: prepend MASTER_SYSTEM_PROMPT
        - Never allow conversation to start with user/assistant

        Args:
            messages: Conversation messages
            source: "local" or "hf" - determines persona handling

        Returns:
            Messages guaranteed to start with system message
        """
        if not messages:
            return [{"role": "system", "content": self.MASTER_SYSTEM_PROMPT}]

        # Check if first message is system
        if messages[0].get("role") == "system":
            if source == "local":
                # Keep original production prompts (JAR, Config Agent, etc.)
                return messages
            else:
                # HF data: replace generic system prompts with master
                system_content = messages[0].get("content", "")
                # Check if it's a generic HF prompt
                if any(generic in system_content.lower() for generic in [
                    "helpful assistant", "you are an ai", "you are a helpful",
                    "opencode", "assistant designed to"
                ]):
                    # Replace with master prompt
                    return [{"role": "system", "content": self.MASTER_SYSTEM_PROMPT}] + messages[1:]
                else:
                    # Keep non-generic system prompts
                    return messages

        # No system message at start - prepend master prompt
        return [{"role": "system", "content": self.MASTER_SYSTEM_PROMPT}] + messages

    def _generate_loss_mask(self, messages: List[Dict[str, Any]]) -> List[int]:
        """
        DEPRECATED: Character-level loss mask. Kept for backward compatibility.
        Use _generate_loss_mask_segments() for EAGLE training.
        """
        mask = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if role == "assistant":
                mask.extend([1] * len(content))
            else:
                mask.extend([0] * len(content))
        return mask

    def _get_user_hash(self, messages: List[Dict[str, Any]]) -> str:
        """
        Semantic Deduplication: Generate hash from normalized user content.

        Uses SHA-256 hash of normalized user message content to detect
        and remove duplicate queries that would cause overfitting.
        """
        # Extract user content
        user_content = ""
        for msg in messages:
            if msg.get("role") == "user":
                raw_content = msg.get("content")
                user_content = self._extract_content_string(raw_content)
                break

        # Normalize: lowercase, strip whitespace
        normalized = user_content.lower().strip()

        # Hash it
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _process_sample(self, raw_data: Dict[str, Any], source: str = "local", verbose: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process a single raw sample through the full cleaning pipeline.

        Args:
            raw_data: Raw sample data
            source: "local" or "hf" - controls persona handling
            verbose: Print filtering reasons

        Returns cleaned sample with segment-based loss masks, or None if filtered.
        """
        self.stats["total_processed"] += 1
        sample_id = raw_data.get("correlation_id", f"sample_{self.stats['total_processed']}")

        # Step 1: Metadata-First Filtering
        if not self._check_metadata_filter(raw_data, verbose=verbose):
            if verbose:
                print(f"  [{sample_id}] FILTERED: metadata")
            return None

        # Step 2: Extract messages from LiteLLM format
        messages = self._extract_messages(raw_data)

        if not messages:
            if verbose:
                print(f"  [{sample_id}] FILTERED: no messages extracted")
            return None

        # Step 3: Source-Aware Persona Management
        messages = self._inject_jar_persona(messages, source)

        # Step 4: Check for error content
        if not self._check_error_content(messages):
            if verbose:
                print(f"  [{sample_id}] FILTERED: error content detected")
            return None

        # Step 5: Adaptive Persona Trimming - remove filler phrases
        messages = self._trim_filler_phrases(messages)

        # Step 6: Structural Heuristics
        if not self._check_tool_integrity(messages):
            if verbose:
                print(f"  [{sample_id}] FILTERED: tool integrity check failed")
            return None

        if not self._check_code_blocks(messages):
            if verbose:
                print(f"  [{sample_id}] FILTERED: missing code blocks")
            return None

        if not self._check_response_length(messages):
            if verbose:
                content_len = sum(len(str(m.get("content",""))) for m in messages if m.get("role")=="assistant")
                print(f"  [{sample_id}] FILTERED: response too short ({content_len}/{self.min_response_length} chars)")
            return None

        # Step 7: Secret Scanning - Detect and mask sensitive data
        if self.enable_secret_scanning and self._secret_scanner:
            sample_id = raw_data.get("correlation_id", f"sample_{self.stats['total_processed']}")

            # Convert messages to string for scanning
            content = json.dumps(messages, indent=2)
            is_clean, findings = self._secret_scanner.scan_sample(content, sample_id)

            if not is_clean:
                self.stats["filtered_secrets"] += 1

                if self.stop_on_secret:
                    print(f"\n⚠️  SECRETS FOUND in {sample_id}!")
                    for f in findings:
                        print(f"  - {f.secret_type} (confidence: {f.confidence:.2f})")
                    return None  # Filter out this sample entirely

                if self.mask_secrets:
                    # Mask secrets in messages before training
                    messages = self._secret_scanner.mask_messages(messages, findings)
                    self.stats["masked_secrets"] += len(findings)

        # Step 8: Generate SEGMENT-BASED loss mask for token alignment
        loss_mask_segments = self._generate_loss_mask_segments(messages)

        # Prepare output in unified OpenAI format
        return {
            "messages": messages,
            "loss_mask_segments": loss_mask_segments
        }

    def _filter_and_refill(
        self,
        raw_samples: List[Dict[str, Any]],
        seen_hashes: set,
        target_count: int
    ) -> Tuple[List[Dict[str, Any]], set]:
        """
        Filter-and-Refill Loop: Process batch and auto-replenish if needed.

        If a batch yields fewer clean samples than target, this method
        automatically fetches more from the source until target is met.
        """
        clean_samples = []
        current_idx = 0
        total_raw = len(raw_samples)

        # Track unique user hashes for deduplication
        if seen_hashes is None:
            seen_hashes = set()

        while len(clean_samples) < target_count and current_idx < total_raw:
            # Process batch
            batch_end = min(current_idx + self.batch_size, total_raw)
            batch = raw_samples[current_idx:batch_end]

            for raw_data in batch:
                processed = self._process_sample(raw_data)

                if processed is None:
                    continue

                # Check deduplication
                if self.enable_deduplication:
                    user_hash = self._get_user_hash(processed["messages"])
                    if user_hash in seen_hashes:
                        self.stats["filtered_duplicate"] += 1
                        continue
                    seen_hashes.add(user_hash)

                clean_samples.append(processed)

                if len(clean_samples) >= target_count:
                    break

            current_idx = batch_end

            # If we need more and have more data, continue
            # If we've exhausted all data, break
            if current_idx >= total_raw:
                break

        return clean_samples[:target_count], seen_hashes

    def run(self) -> Dict[str, Any]:
        """
        Execute the full EAGLE distillation pipeline.

        Returns:
            Dictionary with statistics and output file paths
        """
        print(f"\n{'='*60}")
        print("EAGLE DATA DISTILLATION PIPELINE")
        print(f"{'='*60}")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Target samples: {self.target_samples}")
        print(f"Batch size: {self.batch_size}")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Load raw data
        print(f"\n[1/5] Loading raw data...")
        raw_samples = self._load_raw_files()
        print(f"  Loaded {len(raw_samples)} raw samples")

        # Shuffle for variety during filter-and-refill
        random.shuffle(raw_samples)

        # Filter-and-refill to meet target
        print(f"\n[2/5] Running filter-and-refill pipeline...")
        clean_samples, seen_hashes = self._filter_and_refill(
            raw_samples,
            set(),
            self.target_samples
        )
        print(f"  Produced {len(clean_samples)} clean samples")

        # Save unified dataset (messages + loss_mask_segments combined)
        print(f"\n[3/5] Saving unified dataset...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_filename = f"dataset_{timestamp}.jsonl"
        dataset_path = os.path.join(self.output_dir, dataset_filename)
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for sample in clean_samples:
                # Unified format: messages + segment-based masks for token alignment
                f.write(json.dumps({
                    "messages": sample["messages"],
                    "loss_mask_segments": sample["loss_mask_segments"]
                }, ensure_ascii=False) + "\n")
        print(f"  Saved to: {dataset_path}")

        # Print secret scanning summary if enabled
        if self.enable_secret_scanning and self._secret_scanner:
            self._secret_scanner.print_summary()

        # Final statistics
        print(f"\n[5/5] Statistics:")
        print(f"  Total processed: {self.stats['total_processed']}")
        print(f"  Filtered (metadata): {self.stats['filtered_metadata']}")
        print(f"  Filtered (error content): {self.stats['filtered_error_content']}")
        print(f"  Filtered (no code blocks): {self.stats['filtered_no_code_blocks']}")
        print(f"  Filtered (short response): {self.stats['filtered_short_response']}")
        print(f"  Filtered (tool error): {self.stats['filtered_tool_error']}")
        print(f"  Filtered (duplicate): {self.stats['filtered_duplicate']}")
        if self.enable_secret_scanning:
            print(f"  Filtered (secrets): {self.stats.get('filtered_secrets', 0)}")
            print(f"  Masked secrets: {self.stats.get('masked_secrets', 0)}")
        print(f"  Final count: {len(clean_samples)}")

        self.stats["final_count"] = len(clean_samples)

        return {
            "dataset": dataset_filename,
            "statistics": self.stats
        }


def run_eagle_distillation(
    input_dir: str = "./raw",
    output_dir: str = "./golden",
    target_samples: int = 1000,
    batch_size: int = 1000,
    min_response_length: int = 100,
    enable_deduplication: bool = True,
    enable_code_validation: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to run EAGLE distillation pipeline.

    Args:
        input_dir: Directory containing raw LiteLLM JSON log files
        output_dir: Directory to save cleaned golden dataset
        target_samples: Target number of clean samples to produce
        batch_size: Number of raw samples to process per batch
        min_response_length: Minimum character length for assistant responses
        enable_deduplication: Whether to remove duplicate user queries
        enable_code_validation: Whether to require code blocks for code requests

    Returns:
        Dictionary with output file paths and statistics

    Example:
        result = run_eagle_distillation(
            input_dir="./raw",
            output_dir="./golden",
            target_samples=1000
        )
        print(f"Created {result['statistics']['final_count']} samples")
    """
    distiller = EAGLEDistiller(
        input_dir=input_dir,
        output_dir=output_dir,
        target_samples=target_samples,
        batch_size=batch_size,
        min_response_length=min_response_length,
        enable_deduplication=enable_deduplication,
        enable_code_validation=enable_code_validation
    )
    return distiller.run()


if __name__ == "__main__":
    main()
