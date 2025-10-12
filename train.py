#!/usr/bin/env python3
"""
Simple training script for Smart Product Pricing ML Pipeline
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run main pipeline
from main import main

if __name__ == "__main__":
    main()