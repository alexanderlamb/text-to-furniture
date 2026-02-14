#!/usr/bin/env python3
"""Launch the Text-to-Furniture web UI on http://localhost:8080."""

import sys
from pathlib import Path

# Add src/ to Python path so bare imports work (project convention)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ui.app import main

if __name__ == "__main__":
    main()
