#!/usr/bin/env python3
"""
Train Drafter Script - Entry Point
Usage: python -m p_eagle.scripts.train_drafter --drafter_model ... --feature_dir ...
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..training.trainer import main

if __name__ == "__main__":
    main()
