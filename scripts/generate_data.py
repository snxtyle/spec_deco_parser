import json
import os
import re
import random
import argparse
import subprocess
import shutil
import hashlib
from datetime import datetime
from glob import glob
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

# Try to import psutil for system monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Note: psutil not installed. Run: pip install psutil for detailed system metrics")

# Default paths
DEFAULT_INPUT_DIR = "/Users/suraj.nagre/Desktop/juspay-eval-multilingual/processed"
DEFAULT_OUTPUT_FILE = "/Users/suraj.nagre/Desktop/juspay-eval-multilingual/output/openai_dataset.jsonl"

# HuggingFace dataset config
HF_DATASET_NAME = "Salesforce/xlam-function-calling-60k"


def check_system_resources() -> Dict[str, Any]:
    """Check and return system resource information."""
    resources = {
        "cpu": {},
        "memory": {},
        "storage": {}
    }

    # CPU Information
    if PSUTIL_AVAILABLE:
        resources["cpu"]["count"] = psutil.cpu_count(logical=True)
        resources["cpu"]["physical"] = psutil.cpu_count(logical=False)
        resources["cpu"]["usage_percent"] = psutil.cpu_percent(interval=1)
        # Per-CPU usage
        resources["cpu"]["per_cpu"] = psutil.cpu_percent(interval=1, percpu=True)
    else:
        # Fallback: use subprocess
        try:
            result = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True)
            resources["cpu"]["count"] = int(result.stdout.strip()) if result.returncode == 0 else "N/A"
        except:
            resources["cpu"]["count"] = "N/A"

    # Memory Information
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        resources["memory"]["total_gb"] = round(mem.total / (1024**3), 2)
        resources["memory"]["available_gb"] = round(mem.available / (1024**3), 2)
        resources["memory"]["used_gb"] = round(mem.used / (1024**3), 2)
        resources["memory"]["usage_percent"] = mem.percent
    else:
        resources["memory"]["total_gb"] = "N/A"
        resources["memory"]["available_gb"] = "N/A"
        resources["memory"]["usage_percent"] = "N/A"

    # Storage Information for current directory
    try:
        # Get disk usage for current path
        usage = shutil.disk_usage(".")
        resources["storage"]["total_gb"] = round(usage.total / (1024**3), 2)
        resources["storage"]["used_gb"] = round(usage.used / (1024**3), 2)
        resources["storage"]["free_gb"] = round(usage.free / (1024**3), 2)
        resources["storage"]["usage_percent"] = round((usage.used / usage.total) * 100, 1)
    except Exception as e:
        resources["storage"]["error"] = str(e)

    # Estimate storage needed for output (rough estimate: ~2KB per sample)
    resources["estimate_per_sample_kb"] = 2

    return resources


def print_system_resources(resources: Dict[str, Any]):
    """Print system resources in a formatted way."""
    print(f"\n{'='*60}")
    print(f"SYSTEM RESOURCES")
    print(f"{'='*60}")

    # CPU
    cpu = resources.get("cpu", {})
    print(f"\nCPU:")
    if PSUTIL_AVAILABLE:
        print(f"  Cores: {cpu.get('count', 'N/A')} logical, {cpu.get('physical', 'N/A')} physical")
        print(f"  Usage: {cpu.get('usage_percent', 'N/A')}%")
    else:
        print(f"  Cores: {cpu.get('count', 'N/A')}")
        print(f"  Usage: Install psutil for detailed CPU metrics")

    # Memory
    mem = resources.get("memory", {})
    print(f"\nMemory:")
    if PSUTIL_AVAILABLE:
        print(f"  Total: {mem.get('total_gb', 'N/A')} GB")
        print(f"  Available: {mem.get('available_gb', 'N/A')} GB")
        print(f"  Usage: {mem.get('usage_percent', 'N/A')}%")
    else:
        print(f"  Install psutil for memory details")

    # Storage
    storage = resources.get("storage", {})
    print(f"\nStorage (current disk):")
    if "error" not in storage:
        print(f"  Total: {storage.get('total_gb', 'N/A')} GB")
        print(f"  Free: {storage.get('free_gb', 'N/A')} GB")
        print(f"  Usage: {storage.get('usage_percent', 'N/A')}%")
    else:
        print(f"  Error: {storage.get('error', 'N/A')}")

    est_per_sample = resources.get("estimate_per_sample_kb", 2)
    print(f"\nEstimated output size: ~{est_per_sample} KB per sample")

# Try to import datasets, handle gracefully if not installed
try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not installed. Run: pip install datasets")

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
    # Load .env file if exists
    load_dotenv()
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", os.getenv("HF_TOKEN"))
    if HF_TOKEN:
        print(f"✓ HuggingFace token loaded from .env")
    else:
        HF_TOKEN = None
except ImportError:
    HF_TOKEN = None
    print("Note: python-dotenv not installed. Create .env file for HF token.")


def parse_tool_calls(answers_str: str) -> List[Dict[str, Any]]:
    """Parse tool calls from the answers field."""
    import ast
    try:
        # Try to parse as JSON first
        tool_calls = json.loads(answers_str)
        if isinstance(tool_calls, list):
            return tool_calls
    except:
        pass
    
    # Try parsing as string representation of list
    try:
        tool_calls = ast.literal_eval(answers_str)
        if isinstance(tool_calls, list):
            return tool_calls
    except:
        pass
    
    return []


def convert_xlam_to_messages(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """Convert xlam dataset item to OpenAI messages format with tool_calls."""
    query = item.get("query", "")
    answers = item.get("answers", "")
    tools = item.get("tools", "")
    
    if not query or not answers:
        return None
    
    # Parse tool calls from answers
    tool_calls_data = parse_tool_calls(answers)
    if not tool_calls_data:
        return None
    
    # Parse tools if available
    tools_list = []
    try:
        tools_list = json.loads(tools) if tools else []
    except:
        pass
    
    # Build the conversation in OpenAI format
    messages = [
        {"role": "user", "content": query}
    ]
    
    # Add tool_calls to assistant message
    assistant_tool_calls = []
    for tc in tool_calls_data:
        name = tc.get("name", "")
        arguments = tc.get("arguments", {})
        
        # Convert arguments to JSON string if it's a dict
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
    
    # Add assistant message with tool_calls
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": assistant_tool_calls
    })
    
    return {
        "messages": messages,
        "_source": "huggingface",
        "_dataset": HF_DATASET_NAME,
        "_index": idx
    }


def has_tool_calls(messages: List[Dict[str, Any]]) -> bool:
    """Check if any assistant message in the conversation has tool_calls."""
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
    return False


def convert_to_sharegpt(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI format to ShareGPT format.

    OpenAI format: {"messages": [{"role": "user/assistant/system", "content": "...", "tool_calls": [...]}]}
    ShareGPT format: {"conversations": [{"from": "human/gpt", "value": "..."}]}

    Note: ShareGPT doesn't natively support tool_calls, so we convert them to a string representation.
    """
    messages = sample.get("messages", [])
    if not messages:
        return None

    conversations = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        # Map roles to ShareGPT format
        if role == "system":
            # ShareGPT uses human for system messages too
            from_value = "human"
            value = content or ""
        elif role == "user":
            from_value = "human"
            value = content or ""
        elif role == "assistant":
            from_value = "gpt"
            value = content or ""

            # Convert tool_calls to text representation
            if tool_calls:
                tool_calls_str = "\n[Tool Calls]:\n"
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls_str += f"- {name}({args})\n"
                value = (value or "") + tool_calls_str
        elif role == "tool":
            from_value = "human"  # Tool responses treated as human input context
            value = content or ""
        else:
            continue

        if value:  # Only add non-empty messages
            conversations.append({
                "from": from_value,
                "value": value
            })

    if not conversations:
        return None

    return {"conversations": conversations}


def convert_to_openai(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ShareGPT format to OpenAI format.

    ShareGPT format: {"conversations": [{"from": "human/gpt", "value": "..."}]}
    OpenAI format: {"messages": [{"role": "user/assistant/system", "content": "..."}]}
    """
    conversations = sample.get("conversations", [])
    if not conversations:
        return None

    messages = []

    for conv in conversations:
        from_val = conv.get("from", "")
        value = conv.get("value", "")

        # Map ShareGPT roles to OpenAI roles
        if from_val == "human":
            role = "user"
        elif from_val == "gpt":
            role = "assistant"
        else:
            continue

        if value:
            messages.append({
                "role": role,
                "content": value
            })

    if not messages:
        return None

    return {"messages": messages}


def compute_sample_hash(sample: Dict[str, Any]) -> str:
    """Compute a hash for a sample for deduplication."""
    # Create a normalized representation of the conversation
    messages = sample.get("messages", [])
    content_parts = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])

        # Include role and content
        content_parts.append(f"{role}:{content}")

        # Include tool call names and their arguments
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                content_parts.append(f"tool:{name}:{args}")

    # Create hash
    content_str = "|".join(content_parts)
    return hashlib.sha256(content_str.encode('utf-8')).hexdigest()


def deduplicate_samples(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Remove duplicate samples based on content hash.

    Returns:
        Tuple of (deduplicated samples, stats dict)
    """
    seen_hashes = set()
    unique_samples = []
    stats = {
        "total": len(samples),
        "duplicates": 0,
        "unique": 0
    }

    for sample in samples:
        sample_hash = compute_sample_hash(sample)

        if sample_hash not in seen_hashes:
            seen_hashes.add(sample_hash)
            unique_samples.append(sample)
            stats["unique"] += 1
        else:
            stats["duplicates"] += 1

    return unique_samples, stats


def load_from_existing_output(file_path: str) -> List[Dict[str, Any]]:
    """Load samples from existing output file for resume capability.

    Supports both JSON and JSONL formats.
    """
    samples = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                # JSON array format
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
            else:
                # JSONL format
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            samples.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    except FileNotFoundError:
        print(f"Warning: Resume file not found: {file_path}")
    except Exception as e:
        print(f"Error loading resume file: {e}")

    print(f"Loaded {len(samples)} samples from existing output for resume")
    return samples


def load_from_huggingface(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load data from Salesforce/xlam-function-calling-60k HuggingFace dataset.
    
    Converts to OpenAI format with tool_calls.
    
    Note: This dataset is gated and requires HuggingFace authentication.
    The token is automatically loaded from HUGGINGFACE_TOKEN or HF_TOKEN in .env file.
    """
    if not HF_AVAILABLE:
        print("Error: datasets library not available")
        return []
    
    print(f"Loading dataset: {HF_DATASET_NAME}")
    
    # Check for token
    if not HF_TOKEN:
        print("Warning: No HuggingFace token found in .env file.")
        print("Set HUGGINGFACE_TOKEN or HF_TOKEN in .env for gated datasets.")
        print("Get token from: https://huggingface.co/settings/tokens")
    
    try:
        # Load the dataset with token if available
        dataset = load_dataset(HF_DATASET_NAME, token=HF_TOKEN if HF_TOKEN else False)
        
        # Get the split (usually 'train')
        if isinstance(dataset, dict):
            # Multiple splits - use first available
            split = list(dataset.keys())[0]
            dataset = dataset[split]
        
        print(f"Total samples in dataset: {len(dataset)}")
        
        # Convert all samples to messages format
        valid_samples = []
        for idx, item in enumerate(dataset):
            sample = convert_xlam_to_messages(item, idx)
            
            if sample:
                valid_samples.append(sample)
                
                if limit and len(valid_samples) >= limit:
                    break
        
        print(f"Samples with tool_calls: {len(valid_samples)}")
        return valid_samples
        
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "authentication" in error_msg.lower():
            print("Error: This dataset requires HuggingFace authentication.")
            print("To fix:")
            print("  1. Get a token from: https://huggingface.co/settings/tokens")
            print("  2. Run: huggingface-cli login")
            print("  3. Or add token to the code: load_dataset(..., token='your_token')")
        else:
            print(f"Error loading HuggingFace dataset: {e}")
        return []


def load_from_local(input_dir: str) -> List[Dict[str, Any]]:
    """Load data from local JSON files in the processed directory."""
    files = glob(os.path.join(input_dir, "*.json"))
    
    print(f"Loading {len(files)} files from {input_dir}")
    
    valid_samples = []
    for file_path in files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Get messages
            if isinstance(data, list):
                messages = data
            elif isinstance(data, dict):
                messages = data.get("messages", [])
            else:
                continue
            
            # Check if has tool_calls
            if has_tool_calls(messages):
                sample = {
                    "messages": messages,
                    "_source": "local",
                    "_file": os.path.basename(file_path)
                }
                valid_samples.append(sample)
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    print(f"Local samples with tool_calls: {len(valid_samples)}")
    return valid_samples


# 🔥 Clean assistant content
def clean_content(content):
    if not content or not isinstance(content, str):
        return None

    content = content.strip()

    # 1. remove ```json ... ```
    content = re.sub(r"```json\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL)

    # 2. remove ``` ... ```
    content = re.sub(r"```\s*(.*?)\s*```", r"\1", content, flags=re.DOTALL)

    content = content.strip()

    # 3. try parsing JSON string content
    try:
        parsed = json.loads(content)

        # case: list of structured blocks
        if isinstance(parsed, list):
            texts = []
            for item in parsed:
                if isinstance(item, dict):
                    if "text" in item:
                        texts.append(item["text"])
                    elif "value" in item:
                        texts.append(item["value"])
            if texts:
                return "\n".join(texts)

        # case: dict → convert to string (clean JSON)
        if isinstance(parsed, dict):
            return json.dumps(parsed, indent=2)

    except:
        pass

    return content


def extract_clean_sample(data, file_path=None):
    # 🔥 Handle formats
    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        messages = data.get("messages", [])
    else:
        return None, "invalid_format"

    # nested list
    if len(messages) > 0 and isinstance(messages[0], list):
        messages = messages[0]

    system_msg = None
    user_msg = None
    assistant_msg = None

    # 1. system
    for msg in messages:
        if msg.get("role") == "system":
            system_msg = msg.get("content")
            break

    # 2. user
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content")
            break

    # 3. assistant (final clean)
    for msg in reversed(messages):
        if msg.get("role") == "assistant":

            # skip tool calls
            if msg.get("tool_calls"):
                continue

            content = msg.get("content")
            content = clean_content(content)

            # skip empty
            if not content or content.strip() == "":
                continue

            # skip reasoning chatter
            if content.lower().startswith("i'll") or content.lower().startswith("i will"):
                continue

            assistant_msg = content
            break

    # validation
    if not system_msg:
        return None, "missing_system"

    if not user_msg:
        return None, "missing_user"

    if not assistant_msg:
        return None, "missing_assistant"

    # optional: filter short outputs (important for Medusa)
    if len(assistant_msg.split()) < 30:
        return None, "too_short"

    return {
        "messages": [
            {"role": "system", "content": system_msg.strip()},
            {"role": "user", "content": user_msg.strip()},
            {"role": "assistant", "content": assistant_msg.strip()}
        ]
    }, "valid"


def process_samples(samples: List[Dict[str, Any]], stats: Dict[str, int]):
    """Process samples and extract clean tool-calling conversations."""
    for sample in samples:
        messages = sample.get("messages", [])
        
        if not messages:
            stats["invalid_format"] += 1
            continue
        
        # Build conversation for tool-calling format
        # Keep the full conversation including tool_calls
        conversation = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            
            # Skip empty messages
            if not content and not tool_calls:
                continue
            
            msg_obj = {"role": role}
            
            if content:
                msg_obj["content"] = str(content) if content else ""
            
            # Add tool_calls if present (this is what we want!)
            if tool_calls:
                msg_obj["tool_calls"] = tool_calls
            
            conversation.append(msg_obj)
        
        # Validate we have at least user + assistant (system is optional for HF)
        roles = [m.get("role") for m in conversation]
        
        # Allow both formats: system+user+assistant OR user+assistant
        has_user = "user" in roles
        has_assistant = "assistant" in roles
        
        if not has_user:
            stats["missing_user"] += 1
            continue
        if not has_assistant:
            stats["missing_assistant"] += 1
            continue
        
        # If we have system, keep it. If not, add a generic one for consistency
        if "system" not in roles:
            # For HF samples without system, add a default one
            source = sample.get("_source", "unknown")
            if source == "huggingface":
                # Insert default system message for HF samples
                conversation.insert(0, {
                    "role": "system",
                    "content": "You are a helpful assistant."
                })
        
        # Check that there's at least one tool_call
        has_tool = any(m.get("tool_calls") for m in conversation)
        if not has_tool:
            stats["no_tool_calls"] += 1
            continue
        
        yield {
            "messages": conversation
        }


# ============================================
# Sensitive Data Detection Functions (detect-secrets + custom patterns)
# ============================================

@dataclass
class SecretFinding:
    """Represents a detected sensitive data finding."""
    file_path: str
    line_number: int
    secret_type: str
    severity: str  # HIGH, MEDIUM, LOW
    matched_value: str
    context: str




def find_detect_secrets_cmd() -> Optional[str]:
    """Find the detect-secrets executable path."""
    detect_secrets_paths = [
        "/Users/suraj.nagre/Library/Python/3.9/bin/detect-secrets",
        "/Users/suraj.nagre/Library/Python/3.13/bin/detect-secrets",
        "detect-secrets"
    ]
    
    for path in detect_secrets_paths:
        # Check if command exists
        result = subprocess.run(["which", path.replace("/Users/suraj.nagre/Library/Python/3.9/bin/", "")], 
                              capture_output=True, text=True)
        if result.returncode == 0 or path.startswith("/Users"):
            return path
    
    return "detect-secrets"  # fallback


def install_gitleaks() -> bool:
    """Install gitleaks binary automatically."""
    import platform
    import urllib.request
    import zipfile
    import io

    system = platform.system()
    machine = platform.machine()

    # Determine the correct binary URL
    if system == "Darwin":
        if machine == "arm64":
            url = "https://github.com/zricethezav/gitleaks/releases/download/v8.18.3/gitleaks_8.18.3_darwin_arm64.zip"
        else:
            url = "https://github.com/zricethezav/gitleaks/releases/download/v8.18.3/gitleaks_8.18.3_darwin_x64.zip"
    elif system == "Linux":
        if machine == "x86_64":
            url = "https://github.com/zricethezav/gitleaks/releases/download/v8.18.3/gitleaks_8.18.3_linux_x64.tar.gz"
        else:
            return False
    else:
        return False

    try:
        print("Installing gitleaks...")
        # Download the archive
        response = urllib.request.urlopen(url, timeout=60)
        data = io.BytesIO(response.read())

        # Extract to home bin directory
        import os
        bin_dir = os.path.expanduser("~/Library/Python/3.9/bin")
        os.makedirs(bin_dir, exist_ok=True)

        if url.endswith(".zip"):
            with zipfile.ZipFile(data) as zf:
                zf.extractall(bin_dir)
        else:
            import tarfile
            with tarfile.open(fileobj=data) as tf:
                tf.extractall(bin_dir)

        print(f"✓ gitleaks installed to {bin_dir}")
        return True
    except Exception as e:
        print(f"Failed to install gitleaks: {e}")
        return False


def find_gitleaks_cmd() -> Optional[str]:
    """Find the gitleaks executable path. Auto-installs if not found."""
    gitleaks_paths = [
        "gitleaks",
        os.path.expanduser("~/Library/Python/3.9/bin/gitleaks"),
        "/usr/local/bin/gitleaks",
        "/usr/bin/gitleaks",
        "/opt/homebrew/bin/gitleaks",
    ]

    for path in gitleaks_paths:
        result = subprocess.run(["which", path], capture_output=True, text=True)
        if result.returncode == 0:
            return path

    # Try to auto-install
    if install_gitleaks():
        return os.path.expanduser("~/Library/Python/3.9/bin/gitleaks")

    return None


def find_trufflehog_cmd() -> Optional[str]:
    """Find the trufflehog executable path."""
    # Check if Python trufflehog package is available
    try:
        import trufflehog
        # Return the module path so we can call it
        return "trufflehog"
    except ImportError:
        pass

    # Check for CLI binary
    trufflehog_paths = [
        "trufflehog",
        "/usr/local/bin/trufflehog",
        "/usr/bin/trufflehog",
        "/opt/homebrew/bin/trufflehog",
    ]

    for path in trufflehog_paths:
        result = subprocess.run(["which", path], capture_output=True, text=True)
        if result.returncode == 0:
            return path

    return None


def scan_file_for_secrets(file_path: str) -> List[SecretFinding]:
    """Scan a single file for sensitive data using gitleaks/trufflehog."""
    findings = []

    # Try gitleaks first (faster, comprehensive patterns)
    gitleaks_cmd = find_gitleaks_cmd()

    if gitleaks_cmd:
        try:
            # Use gitleaks to scan the file
            result = subprocess.run(
                [gitleaks_cmd, "detect", "--source", file_path, "--no-git", "--report-format", "json"],
                capture_output=True,
                text=True,
                timeout=60
            )

            # Parse gitleaks JSON output
            if result.stdout.strip():
                try:
                    # Try to parse each line as JSON (gitleaks outputs one JSON per line)
                    for line in result.stdout.strip().split('\n'):
                        if not line.strip():
                            continue
                        try:
                            secret = json.loads(line)
                            secret_type = secret.get("RuleID", secret.get("MatchRuleID", "Unknown"))
                            line_num = secret.get("StartLine", secret.get("LineNumber", 1))

                            # Determine severity based on rule ID
                            severity = "HIGH"
                            if any(x in secret_type.lower() for x in ["test", "dev", "sample"]):
                                severity = "MEDIUM"

                            # Mask the detected secret
                            match = secret.get("Match", "")
                            if match and len(match) > 12:
                                masked = match[:4] + "*" * (len(match) - 8) + match[-4:]
                            else:
                                masked = "[REDACTED]"

                            findings.append(SecretFinding(
                                file_path=file_path,
                                line_number=line_num,
                                secret_type=secret_type,
                                severity=severity,
                                matched_value=masked,
                                context=secret.get("Context", "")[:200]
                            ))
                        except json.JSONDecodeError:
                            continue
                except Exception:
                    pass
        except Exception:
            pass

    # If gitleaks not available, try trufflehog
    if not findings:
        trufflehog_cmd = find_trufflehog_cmd()

        if trufflehog_cmd:
            try:
                # Check if it's the Python module or CLI binary
                if trufflehog_cmd == "trufflehog" and not shutil.which("trufflehog"):
                    # Use Python module
                    try:
                        import trufflehog
                        # Run as Python module
                        result = subprocess.run(
                            ["python3", "-m", "trufflehog", "filesystem", file_path, "--json"],
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                    except ImportError:
                        result = subprocess.CompletedPipe()
                        result.returncode = 1
                else:
                    # Use CLI binary
                    result = subprocess.run(
                        [trufflehog_cmd, "filesystem", file_path, "--json"],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )

                if result.stdout.strip():
                    for line in result.stdout.strip().split('\n'):
                        if not line.strip():
                            continue
                        try:
                            secret = json.loads(line)
                            secret_type = secret.get("DetectorType", secret.get("Source", {}).get("type", "Unknown"))
                            line_num = secret.get("Source", {}).get("line", 1)

                            severity = "HIGH"
                            if any(x in secret_type.lower() for x in ["test", "dev", "sample"]):
                                severity = "MEDIUM"

                            findings.append(SecretFinding(
                                file_path=file_path,
                                line_number=line_num,
                                secret_type=secret_type,
                                severity=severity,
                                matched_value="[REDACTED]",
                                context=secret.get("Source", {}).get("context", "")[:200]
                            ))
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass

    # Fallback: try detect-secrets if neither gitleaks nor trufflehog available
    if not findings:
        detect_secrets_cmd = find_detect_secrets_cmd()

        try:
            result = subprocess.run(
                [detect_secrets_cmd, "scan", file_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                output = json.loads(result.stdout)
                results = output.get("results", {})

                if results:
                    file_results = results.get(file_path, [])

                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.read().split('\n')

                    for secret in file_results:
                        line_num = secret.get("line_number", 1)
                        secret_type = secret.get("type", "Unknown")

                        severity = "HIGH"
                        if "High Entropy" in secret_type:
                            severity = "MEDIUM"

                        masked = f"[{secret_type}]"

                        context_start = max(0, line_num - 2)
                        context_end = min(len(lines), line_num + 2)
                        context = "\n".join(lines[context_start:context_end])

                        findings.append(SecretFinding(
                            file_path=file_path,
                            line_number=line_num,
                            secret_type=secret_type,
                            severity=severity,
                            matched_value=masked,
                            context=context[:200]
                        ))
        except Exception:
            pass

    return findings


def scan_directory(directory: str, extensions: List[str] = None) -> List[SecretFinding]:
    """Scan a directory for sensitive data in files with specified extensions."""
    if extensions is None:
        extensions = ['.json', '.yaml', '.yml', '.env', '.txt', '.py', '.js', '.ts', '.sh', '.tf', '.cfg', '.ini', '.properties']
    
    all_findings = []
    
    # Walk through directory
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common non-relevant paths
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', '.git']]
        
        for file in files:
            # Check file extension
            if not any(file.endswith(ext) for ext in extensions):
                continue
            
            file_path = os.path.join(root, file)
            
            # Skip the output directory
            if 'output' in file_path or 'node_modules' in file_path:
                continue
            
            findings = scan_file_for_secrets(file_path)
            all_findings.extend(findings)
    
    return all_findings


def scan_gcs_bucket(bucket_name: str, prefix: str = "") -> List[SecretFinding]:
    """Scan a GCS bucket for sensitive data.
    
    Requires gsutil or google-cloud-storage package.
    """
    findings = []
    
    # Try using gsutil first
    try:
        cmd = ["gsutil", "ls", f"gs://{bucket_name}/{prefix}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"Error listing GCS bucket: {result.stderr}")
            return findings
        
        files = result.stdout.strip().split('\n')
        
        for file_path in files:
            if not file_path or not file_path.startswith('gs://'):
                continue
            
            # Download and scan each file
            try:
                # Use gsutil to copy to temp location
                temp_file = "/tmp/gcs_scan_temp"
                subprocess.run(["gsutil", "cp", file_path, temp_file], 
                             capture_output=True, timeout=60)
                
                file_findings = scan_file_for_secrets(temp_file)
                
                # Update file paths to reflect GCS
                for finding in file_findings:
                    finding.file_path = file_path
                
                findings.extend(file_findings)
                
                # Cleanup
                try:
                    os.remove(temp_file)
                except:
                    pass
                    
            except Exception as e:
                print(f"Error scanning {file_path}: {e}")
                
    except FileNotFoundError:
        # Try google-cloud-storage package
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            blobs = bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                if blob.name.endswith('/'):  # Skip directories
                    continue
                
                try:
                    # Save to temp file and scan with detect-secrets
                    content = blob.download_as_text()
                    temp_file = "/tmp/gcs_scan_blob_temp"
                    with open(temp_file, 'w') as f:
                        f.write(content)
                    
                    file_findings = scan_file_for_secrets(temp_file)
                    
                    # Update file paths
                    for finding in file_findings:
                        finding.file_path = f"gs://{bucket_name}/{blob.name}"
                    
                    findings.extend(file_findings)
                    
                    # Cleanup
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                                
                except Exception as e:
                    print(f"Error scanning {blob.name}: {e}")
                    
        except ImportError:
            print("Error: google-cloud-storage not installed.")
            print("Install with: pip install google-cloud-storage")
            print("Or use gsutil: https://cloud.google.com/storage/docs/gsutil_install")
    
    return findings


def print_findings(findings: List[SecretFinding]):
    """Print findings in a formatted way."""
    if not findings:
        print("\n✓ No sensitive data found!")
        return
    
    # Group by severity
    high = [f for f in findings if f.severity == "HIGH"]
    medium = [f for f in findings if f.severity == "MEDIUM"]
    low = [f for f in findings if f.severity == "LOW"]
    
    print(f"\n{'='*60}")
    print(f"SENSITIVE DATA DETECTED")
    print(f"{'='*60}")
    print(f"\nTotal findings: {len(findings)}")
    print(f"  - HIGH severity: {len(high)}")
    print(f"  - MEDIUM severity: {len(medium)}")
    print(f"  - LOW severity: {len(low)}")
    
    if high:
        print(f"\n{'='*60}")
        print(f"HIGH SEVERITY (Immediate Action Required)")
        print(f"{'='*60}")
        for f in high:
            print(f"\n  File: {f.file_path}")
            print(f"  Line: {f.line_number}")
            print(f"  Type: {f.secret_type}")
            print(f"  Value: {f.matched_value}")
            print(f"  Context: {f.context[:100]}...")
    
    if medium:
        print(f"\n{'='*60}")
        print(f"MEDIUM SEVERITY (Review Recommended)")
        print(f"{'='*60}")
        for f in medium[:5]:  # Show first 5
            print(f"\n  File: {f.file_path}")
            print(f"  Type: {f.secret_type}")
            print(f"  Value: {f.matched_value}")
    
    if low:
        print(f"\n{'='*60}")
        print(f"LOW SEVERITY (Informational)")
        print(f"{'='*60}")
        print(f"  Found {len(low)} low severity items")
        print(f"  Run with --verbose to see details")


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation dataset from multiple sources"
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

    # Required: number of samples
    parser.add_argument(
        "-n", "--num-samples",
        type=int,
        default=1000,
        help="Total number of data points to generate"
    )

    # Source selection (at least one required)
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
    parser.add_argument(
        "--gcs",
        type=str,
        default=None,
        help="Use GCS bucket (e.g., my-bucket)"
    )

    # Ratio (required when multiple sources, format: "hf:0.6,local:0.4")
    parser.add_argument(
        "--ratio",
        type=str,
        default=None,
        help="Ratio for mixing sources (e.g., hf:0.6,local:0.4). Required when using hf+local or hf+gcs."
    )

    # GCS options
    parser.add_argument(
        "--gcs-prefix",
        type=str,
        default="",
        help="GCS prefix to scan"
    )

    # Other options
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Input directory for local files"
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=HF_DATASET_NAME,
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=0,
        help="Minimum words in assistant response (0 = no filter)"
    )
    parser.add_argument(
        "--stop-on-secret",
        action="store_true",
        help="Stop if secrets found (default: continue and report)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing output file (specify path to JSONL/JSON file)"
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

    # Secret scanning always enabled
    scan_secrets = True

    # Output always goes to ./output
    
    # Subcommand for standalone secret scanning
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Scan-secrets command
    scan_parser = subparsers.add_parser("scan-secrets", help="Scan for sensitive data")
    scan_parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Path to scan (directory or GCS bucket)"
    )
    scan_parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=['.json', '.yaml', '.yml', '.env', '.txt', '.py', '.js', '.ts', '.sh', '.tf', '.cfg', '.ini', '.properties'],
        help="File extensions to scan"
    )
    scan_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output findings to JSON file"
    )
    
    args = parser.parse_args()

    # Check and display system resources
    resources = check_system_resources()
    print_system_resources(resources)

    # Estimate required storage
    estimated_storage_mb = (args.num_samples * resources.get("estimate_per_sample_kb", 2)) / 1024
    free_storage_gb = resources.get("storage", {}).get("free_gb", 0)

    if estimated_storage_mb > free_storage_gb * 1024:
        print(f"\n⚠️  Warning: Estimated output ({estimated_storage_mb:.1f} MB) exceeds free storage ({free_storage_gb:.1f} GB)")

    # Handle scan-secrets command
    if args.command == "scan-secrets":
        print(f"\n{'='*60}")
        print(f"SCANNING FOR SENSITIVE DATA")
        print(f"{'='*60}")

        # Clean bucket name if provided with gs:// prefix
        path = args.path
        if path.startswith("gs://"):
            path = path[5:]

        if path and not os.path.exists(path):
            print(f"\nScanning GCS bucket: {path}")
            findings = scan_gcs_bucket(path, "")
        else:
            print(f"\nScanning directory: {args.path}")
            findings = scan_directory(args.path, args.extensions)

        print_findings(findings)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump([{
                    "file_path": finding.file_path,
                    "line_number": finding.line_number,
                    "secret_type": finding.secret_type,
                    "severity": finding.severity,
                    "matched_value": finding.matched_value,
                    "context": finding.context
                } for finding in findings], f, indent=2)
            print(f"\nFindings saved to: {args.output}")

        return

    # ============================================
    # Data Generation (main functionality)
    # ============================================

    # Determine sources
    sources = []
    if args.hf:
        sources.append("hf")
    if args.local:
        sources.append("local")
    if args.gcs:
        sources.append("gcs")

    if not sources:
        print("Error: Must specify at least one source (--hf, --local, or --gcs)")
        return

    # Validate: can't use local + gcs together (only hf can mix with either)
    has_local = "local" in sources
    has_gcs = "gcs" in sources
    if has_local and has_gcs:
        print("Error: Cannot use --local and --gcs together. Use --hf to combine with either.")
        return

    # Ratio handling
    ratio = None
    if len(sources) > 1:
        if not args.ratio:
            print(f"Error: --ratio required when using multiple sources")
            print(f"Example: --ratio hf:0.6,local:0.4")
            return

        # Parse ratio
        ratio = {}
        for part in args.ratio.split(","):
            part = part.strip()
            if ":" in part:
                key, value = part.split(":", 1)
                ratio[key.strip()] = float(value.strip())

        # Validate ratio
        total = sum(ratio.values())
        if abs(total - 1.0) > 0.01:
            print(f"Warning: Ratio values sum to {total}, normalizing")
            ratio = {k: v / total for k, v in ratio.items()}

        # Verify all sources in ratio
        for src in sources:
            if src not in ratio:
                print(f"Error: Source '{src}' not in ratio. Include all sources.")
                return
    else:
        # Single source
        ratio = {sources[0]: 1.0}

    # GCS bucket name
    if args.gcs:
        bucket_name = args.gcs
        if bucket_name.startswith("gs://"):
            bucket_name = bucket_name[5:]

    print(f"\n{'='*60}")
    print(f"GENERATING DATASET")
    print(f"{'='*60}")
    print(f"Total samples: {args.num_samples}")
    print(f"Sources: {', '.join(sources)}")
    if len(sources) > 1:
        print(f"Ratio: {args.ratio}")
    print(f"Format: {args.format}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output, f"dataset_{timestamp}.jsonl")
    print(f"Output: {output_file}")

    # Load samples from each source
    all_samples = {"hf": [], "local": [], "gcs": []}

    if "hf" in sources:
        if not HF_AVAILABLE:
            print("Error: HuggingFace not available. Install: pip install datasets")
            return
        print(f"\nLoading from HuggingFace: {args.hf_dataset}")
        all_samples["hf"] = load_from_huggingface()
        print(f"  Loaded {len(all_samples['hf'])} samples")

    if "local" in sources:
        print(f"\nLoading from local: {args.input_dir}")
        all_samples["local"] = load_from_local(args.input_dir)
        print(f"  Loaded {len(all_samples['local'])} samples")

    if "gcs" in sources:
        print(f"\nLoading from GCS: {bucket_name}")
        # GCS loading - placeholder
        all_samples["gcs"] = []
        print(f"  Loaded {len(all_samples['gcs'])} samples")

    # Calculate target per source
    target_per_source = {src: int(args.num_samples * ratio[src]) for src in sources}
    print(f"\nTarget per source: {target_per_source}")

    # Process and sample from each source
    processed_samples = {"hf": [], "local": [], "gcs": []}
    stats = {
        "missing_system": 0,
        "missing_user": 0,
        "missing_assistant": 0,
        "invalid_format": 0,
        "too_short": 0,
        "no_tool_calls": 0
    }

    for source, samples in all_samples.items():
        if not samples:
            continue
        source_processed = list(process_samples(samples, stats))
        processed_samples[source] = source_processed

    # Combine samples based on ratio
    final_samples = []
    for source, count in target_per_source.items():
        source_samples = processed_samples.get(source, [])
        if len(source_samples) >= count:
            random.shuffle(source_samples)
            final_samples.extend(source_samples[:count])
        else:
            print(f"  Warning: Only {len(source_samples)} available for {source}, using all")
            final_samples.extend(source_samples)

    # Deduplicate samples if requested
    if args.deduplicate:
        print(f"\nDeduplicating {len(final_samples)} samples...")
        final_samples, dedup_stats = deduplicate_samples(final_samples)
        print(f"  Removed {dedup_stats['duplicates']} duplicates")
        print(f"  Unique samples: {dedup_stats['unique']}")

    # Handle resume capability
    existing_samples = []
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        existing_samples = load_from_existing_output(args.resume)
        if existing_samples:
            # Get hashes of existing samples
            existing_hashes = set()
            for sample in existing_samples:
                h = compute_sample_hash(sample)
                existing_hashes.add(h)

            # Filter out duplicates with existing samples
            new_samples = []
            resume_duplicates = 0
            for sample in final_samples:
                h = compute_sample_hash(sample)
                if h not in existing_hashes:
                    new_samples.append(sample)
                else:
                    resume_duplicates += 1

            print(f"  Skipping {resume_duplicates} samples already in resume file")
            final_samples = existing_samples + new_samples
            print(f"  Total samples after resume: {len(final_samples)}")

    # Limit to num_samples
    final_samples = final_samples[:args.num_samples]

    print(f"\nTotal samples to write: {len(final_samples)}")

    # Write to output file (JSON or JSONL)
    count_valid = 0
    secret_findings = []
    temp_file = "/tmp/secret_scan_temp"

    if args.output_format == "json":
        # JSON format - collect all samples first
        output_data = []

        with open(output_file.replace('.jsonl', '.json'), "w") as out_f:
            for idx, sample in enumerate(final_samples):
                # Apply minimum word filter if specified
                if args.min_words > 0:
                    assistant_content = None
                    for msg in sample.get("messages", []):
                        if msg.get("role") == "assistant" and msg.get("content"):
                            assistant_content = msg.get("content", "")
                            break

                    if assistant_content and len(assistant_content.split()) < args.min_words:
                        stats["too_short"] += 1
                        continue

                # Real-time secret scanning (always scan the original sample)
                if scan_secrets:
                    # Write sample to temp file for scanning
                    with open(temp_file, "w") as tf:
                        sample_str = json.dumps(sample, ensure_ascii=False)
                        tf.write(sample_str)

                    # Scan the temp file
                    findings = scan_file_for_secrets(temp_file)

                    if findings:
                        for f in findings:
                            f.file_path = f"sample_{idx}"
                            secret_findings.append(f)

                        if args.stop_on_secret:
                            print(f"\n⚠️  SECRETS FOUND in sample {idx}!")
                            for f in findings[:3]:
                                print(f"  - {f.secret_type} (line {f.line_number})")
                            print("\nStopping due to --stop-on-secret flag")
                            break

                # Apply format conversion if needed
                if args.format == "sharegpt":
                    output_sample = convert_to_sharegpt(sample)
                else:
                    output_sample = sample

                if output_sample:
                    output_data.append(output_sample)
                    count_valid += 1

                # Progress indicator for large datasets
                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1}/{len(final_samples)} samples...")

            # Write all data as JSON array
            json.dump(output_data, out_f, ensure_ascii=False, indent=2)
            output_file = output_file.replace('.jsonl', '.json')

        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

        print("\n===== SUMMARY =====")
        print(f"Total samples: {len(final_samples)}")
        print(f"Valid samples: {count_valid}")

        print("\nDrop reasons:")
        for k, v in stats.items():
            if v > 0:
                print(f"  {k}: {v}")

        # Secret scanning summary
        if scan_secrets:
            print("\n===== SECRET SCAN SUMMARY =====")
            if secret_findings:
                high_secrets = [f for f in secret_findings if f.severity == "HIGH"]
                medium_secrets = [f for f in secret_findings if f.severity == "MEDIUM"]
                print(f"Secrets found: {len(secret_findings)}")
                print(f"  - HIGH severity: {len(high_secrets)}")
                print(f"  - MEDIUM severity: {len(medium_secrets)}")

                if args.stop_on_secret:
                    print("\n⚠️  Processing stopped early due to secret detection!")
            else:
                print("✓ No secrets detected in any samples!")

        print(f"\nOutput written to: {output_file}")
        return

    # JSONL format (default)
    with open(output_file, "w") as out_f:
        for idx, sample in enumerate(final_samples):
            # Apply minimum word filter if specified
            if args.min_words > 0:
                assistant_content = None
                for msg in sample.get("messages", []):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        assistant_content = msg.get("content", "")
                        break

                if assistant_content and len(assistant_content.split()) < args.min_words:
                    stats["too_short"] += 1
                    continue

            # Real-time secret scanning (always scan the original sample)
            if scan_secrets:
                # Write sample to temp file for scanning
                with open(temp_file, "w") as tf:
                    sample_str = json.dumps(sample, ensure_ascii=False)
                    tf.write(sample_str)

                # Scan the temp file
                findings = scan_file_for_secrets(temp_file)

                if findings:
                    for f in findings:
                        f.file_path = f"sample_{idx}"
                        secret_findings.append(f)

                    if args.stop_on_secret:
                        print(f"\n⚠️  SECRETS FOUND in sample {idx}!")
                        for f in findings[:3]:
                            print(f"  - {f.secret_type} (line {f.line_number})")
                        print("\nStopping due to --stop-on-secret flag")
                        break

            # Apply format conversion if needed
            if args.format == "sharegpt":
                output_sample = convert_to_sharegpt(sample)
            else:
                output_sample = sample

            if output_sample:
                out_f.write(json.dumps(output_sample, ensure_ascii=False) + "\n")
                count_valid += 1

            # Progress indicator for large datasets
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(final_samples)} samples...")

    # Clean up temp file
    if os.path.exists(temp_file):
        os.remove(temp_file)

    print("\n===== SUMMARY =====")
    print(f"Total samples: {len(final_samples)}")
    print(f"Valid samples: {count_valid}")

    print("\nDrop reasons:")
    for k, v in stats.items():
        if v > 0:
            print(f"  {k}: {v}")

    # Secret scanning summary
    if scan_secrets:
        print("\n===== SECRET SCAN SUMMARY =====")
        if secret_findings:
            high_secrets = [f for f in secret_findings if f.severity == "HIGH"]
            medium_secrets = [f for f in secret_findings if f.severity == "MEDIUM"]
            print(f"Secrets found: {len(secret_findings)}")
            print(f"  - HIGH severity: {len(high_secrets)}")
            print(f"  - MEDIUM severity: {len(medium_secrets)}")

            if args.stop_on_secret:
                print("\n⚠️  Processing stopped early due to secret detection!")
        else:
            print("✓ No secrets detected in any samples!")

    print(f"\nOutput written to: {output_file}")


if __name__ == "__main__":
    main()
