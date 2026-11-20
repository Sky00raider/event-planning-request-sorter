"""One command for a fresh machine or a Colab GPU: train, evaluate, render the demo.

    python scripts/run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for step in (["-m", "app.cli", "train"], ["scripts/evaluate.py"], ["scripts/render_demo.py"]):
    print("\n>>", " ".join(step), flush=True)
    subprocess.run([sys.executable, *step], cwd=ROOT, check=True)
