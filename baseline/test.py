#!/usr/bin/env python3
from pathlib import Path
import subprocess
import argparse

FORMAT = "mp4" # Change this to the video format you are using (e.g., "avi", "mp4", etc.)

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Directory to count files in")
parser.add_argument("--iterations", type=int, default=None,
                    help="Override number of runs (otherwise = files//2)")
args = parser.parse_args()

dir_path = Path(args.directory)

# Count only files (not directories)
num_files = sum(1 for f in dir_path.iterdir() if f.is_file())

runs = args.iterations if args.iterations is not None else num_files // 2

print(f"Found {num_files} files → running baseline.py {runs} times")

for i in range(runs):
    print(f"Run {i+1}/{runs}")
    subprocess.run(["python3", "baseline.py", f"{dir_path}/{i}.{FORMAT}"], check=True)