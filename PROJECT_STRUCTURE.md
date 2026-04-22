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
├── checkpoints/                  # Model checkpoints
│   ├── drafter/                  # Drafter checkpoints
│   └── target/                   # Target model configs
│
├── logs/                         # Training & inference logs
│
├── scripts/                      # Standalone utility scripts
│   ├── generate_data.py          # Main data generation script
│   └── scan_secrets.py           # Secret scanning utility
│
├── p_eagle/                      # Main Python package
│   ├── __init__.py
│   ├── README.md                 # Package documentation
│   │
│   ├── data_preparation/         # Data preparation module
│   │   ├── __init__.py
│   │   ├── data_manager.py       # EAGLEDistiller class
│   │   ├── secret_scanner.py     # Secret detection
│   │   └── filters.py            # Data filtering logic
│   │
│   ├── models/                   # Neural network models
│   │   ├── __init__.py
│   │   ├── eagle_drafter.py      # EagleDrafterModel
│   │   ├── mtp_heads.py          # Multi-Token Prediction heads
│   │   └── tree_attention.py     # Tree attention mechanisms
│   │
│   ├── training/                 # Training modules
│   │   ├── __init__.py
│   │   ├── feature_extractor.py  # Extract features from target
│   │   ├── trainer.py            # Drafter training loop
│   │   └── optimizers.py         # Custom optimizers
│   │
│   ├── inference/                # Inference engine
│   │   ├── __init__.py
│   │   ├── inference_engine.py   # PEAGLEInference class
│   │   ├── speculative_decoder.py # Speculative decoding logic
│   │   └── verifier.py           # Token verification
│   │
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   ├── feature_utils.py      # Feature manipulation
│   │   ├── loss_utils.py         # Loss functions
│   │   ├── metrics.py            # Evaluation metrics
│   │   └── io_utils.py           # File I/O utilities
│   │
│   └── scripts/                  # Entry point scripts
│       ├── extract_features.py   # Feature extraction CLI
│       ├── train_drafter.py      # Training CLI
│       └── run_inference.py      # Inference CLI
│
├── configs/                      # Configuration files
│   ├── extract_features.yaml
│   ├── train_drafter.yaml
│   └── inference.yaml
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_data_preparation.py
│   ├── test_models.py
│   └── test_inference.py
│
└── docs/                         # Documentation
    ├── INSTALL.md
    ├── USAGE.md
    ├── ARCHITECTURE.md
    └── FAQ.md
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
