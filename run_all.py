"""Runs the complete pipeline in order and writes results/run_log.txt.

    python run_all.py --repo https://github.com/<username>/<repo-name>
    python run_all.py --quick      (1-epoch smoke test, only to check that everything runs)
"""
import os
import subprocess
import sys

from common import RESULTS, ROOT

args = sys.argv[1:]
quick = "--quick" in args
repo = args[args.index("--repo") + 1] if "--repo" in args else None

steps = [("train_baseline.py", True), ("ptq.py", False), ("qat.py", True), ("prune_unstructured.py", True),
         ("prune_structured.py", True), ("export_onnx.py", False), ("benchmark.py", True), ("make_report.py", False)]

with open(os.path.join(RESULTS, "run_log.txt"), "w") as log:
    for script, takes_quick in steps:
        cmd = [sys.executable, os.path.join(ROOT, script)]
        if quick and takes_quick:
            cmd.append("--quick")
        if script == "make_report.py" and repo:
            cmd += ["--repo", repo]
        header = f"\n===== {script} =====\n"
        print(header, end="")
        log.write(header)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        if proc.wait():
            sys.exit(f"{script} failed - see the messages above.")
print("\nAll steps finished.")
