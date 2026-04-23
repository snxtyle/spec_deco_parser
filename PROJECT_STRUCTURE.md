# P-EAGLE Project Structure

```
juspay-eval-multilingual/
│
├── README.md                     # Project overview
├── PROJECT_STRUCTURE.md          # This file
├── setup.py                      # Package installation
├── requirements.txt              # Python dependencies
│
├── data/                         # Data directory (gitignored)
│   ├── raw/                      # Raw LiteLLM logs
│   ├── processed/                # Processed JSON files
│   ├── features/                 # Extracted features (.pt files)
│   └── output/                   # Generated datasets
│
├── scripts/                      # Standalone utility scripts
│   └── generate_data.py          # Main data generation script
│
├── p_eagle/                      # Main Python package
│   ├── __init__.py
│   │
│   ├── data_preparation/         # Data preparation module
│   │   ├── __init__.py
│   │   └── data_manager.py       # EAGLEDistiller, SmartSecretScanner
│   │
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   ├── eagle_drafter.py      # EagleDrafterModel, EagleMTPHead
│   │   └── tree_attention.py     # Tree attention mechanisms
│   │
│   ├── training/                 # Training modules
│   │   ├── __init__.py
│   │   ├── feature_extractor.py  # Tri-layer feature extraction
│   │   └── trainer.py            # Drafter training loop
│   │
│   ├── inference/                # Inference engine
│   │   ├── __init__.py
│   │   └── inference_engine.py   # PEAGLEInference class
│   │
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── feature_utils.py      # Feature manipulation
│   │   ├── loss_utils.py         # Loss functions
│   │   └── metrics.py            # Evaluation metrics
│   │
│   └── scripts/                  # Entry point scripts
│       ├── extract_features.py   # Feature extraction CLI
│       ├── train_drafter.py      # Training CLI
│       └── run_inference.py      # Inference CLI
```

## File Locations

### Core Application Files

| File | Purpose | Location |
|------|---------|----------|
| `data_manager.py` | EAGLE data distillation | `p_eagle/data_preparation/` |
| `generate_data.py` | CLI for data generation | `scripts/` |
| `eagle_drafter.py` | Neural network models | `p_eagle/models/` |
| `trainer.py` | Training loop | `p_eagle/training/` |
| `inference_engine.py` | Inference | `p_eagle/inference/` |

### Data Files

| File Type | Location |
|-----------|----------|
| Raw LiteLLM logs | `data/raw/` |
| Processed JSON | `data/processed/` |
| Features (.pt) | `data/features/` |
| Output datasets | `data/output/` |

### Model Files

| File Type | Location |
|-----------|----------|
| Drafter checkpoints | `checkpoints/drafter/` |
| Training logs | `logs/training/` |
| Inference logs | `logs/inference/` |
