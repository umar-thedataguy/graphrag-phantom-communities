"""Run the full audit end to end. Takes ~15-20 min on a laptop CPU."""
import subprocess
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
for script in ("run_control.py", "run_planted.py", "run_corpora.py",
               "make_figures.py"):
    print(f"\n===== {script} =====")
    rc = subprocess.call([sys.executable, str(HERE / script)])
    if rc != 0:
        sys.exit(f"{script} failed with code {rc}")
print("\nAll experiments complete. See results/ for JSON and figures.")
