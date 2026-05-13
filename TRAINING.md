# P-EAGLE Training Guide

## Quick Start

### Single GPU Training
```bash
./automation.sh single
```

### Multi-GPU Training (2x DGX Spark)

**Prerequisites:**
1. Clone repo on both machines
2. Sync data to worker:
   ```bash
   ./scripts/sync_to_worker.sh 192.168.1.105
   ```

**Launch:**
```bash
# On Master (this machine):
./automation.sh multi 192.168.1.100

# On Worker (.105 machine):
./automation.sh multi 192.168.1.100
```

## Configuration

Edit these in `automation.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPECULATION_DEPTH` | 2 | Number of future tokens to predict |
| `MAX_SEQ_LEN` | 1024 | Tokens per sequence (reduce if OOM) |
| `BATCH_SIZE` | 1 | Per-GPU batch size |
| `NUM_EPOCHS` | 3 | Training epochs |

## Memory Issues?

Reduce these values in `automation.sh`:
```bash
SPECULATION_DEPTH=1      # Instead of 2
MAX_SEQ_LEN=512          # Instead of 1024
```

## What Was Fixed

1. **Memory**: Removed slow `.tolist()` conversion in window selection
2. **Multi-GPU**: Added DDP support for training across 2x DGX Spark
3. **Unified Script**: Single `automation.sh` handles both modes
