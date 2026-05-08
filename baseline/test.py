#!/usr/bin/env python3
from pathlib import Path
import subprocess
import argparse
import zipfile
import os

FORMAT = "avi" # Change this to the video format you are using (e.g., "avi", "mp4", etc.)

DS_EVAL = "../pytorch/datasets/ds-eval.zip"
DS_DEST = "/tmp"

os.makedirs(DS_DEST, exist_ok=True)
with zipfile.ZipFile(DS_EVAL, 'r') as zip_ref:
    zip_ref.extractall(DS_DEST)

# parser = argparse.ArgumentParser()
# parser.add_argument("directory", help="Directory to count files in")
# parser.add_argument("--iterations", type=int, default=None,
#                     help="Override number of runs (otherwise = files//2)")
# args = parser.parse_args()

# dir_path = Path(args.directory)
dir_path = Path(f"{DS_DEST}/ds")

# Count only files (not directories)
num_files = sum(1 for f in dir_path.iterdir() if f.is_file())

# runs = args.iterations if args.iterations is not None else num_files // 2
runs = num_files // 2

print(f"Found {num_files} files → running baseline.py {runs} times")

for i in range(runs):
    print(f"Run {i+1}/{runs}")
    subprocess.run(["python3", "baseline.py", f"{dir_path}/{i}.{FORMAT}"], check=True)
