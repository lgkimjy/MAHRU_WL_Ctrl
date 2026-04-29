import os
import runpy
import sys
from pathlib import Path


logs_dir = Path(__file__).resolve().parent
repo_root = logs_dir.parent

if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    log_arg = Path(sys.argv[1])
    if not log_arg.is_absolute():
        from_logs = logs_dir / log_arg
        from_root = repo_root / log_arg
        if from_logs.exists() and not from_root.exists():
            sys.argv[1] = str(from_logs)

os.chdir(repo_root)
sys.path.insert(0, str(repo_root))
runpy.run_path(str(repo_root / "reaction_wheel_ref_eval.py"), run_name="__main__")
