#!/usr/bin/env python3
"""
Inference Script - Entry Point
Usage: python -m p_eagle.scripts.run_inference --target_model ... --drafter_checkpoint ...
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ..inference.inference_engine import main

if __name__ == "__main__":
    main()
