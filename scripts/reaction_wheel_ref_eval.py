import os
import runpy
import sys
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
os.chdir(repo_root)
sys.path.insert(0, str(repo_root))
runpy.run_path(str(repo_root / "reaction_wheel_ref_eval.py"), run_name="__main__")
