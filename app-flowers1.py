"""Compatibility entry point for the flower classifier training script."""
import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("app-flowers.py")), run_name="__main")
