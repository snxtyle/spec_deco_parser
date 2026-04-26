#!/usr/bin/env python3
"""
Feature Extraction Script - Entry Point
Usage: python -m p_eagle.scripts.extract_features --model_path ... --input_data ...
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..training.feature_extractor import main

if __name__ == "__main__":
    main()
