import tcn_curve
import zipfile
import os

DS_TRAIN = "./datasets/ds-eval.zip"
DS_DEST = "/tmp"
OUTPUT_DIR = "./output/tcn_eidx"

os.makedirs(DS_DEST, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/curves", exist_ok=True)
with zipfile.ZipFile(DS_TRAIN, 'r') as zip_ref:
    zip_ref.extractall(DS_DEST)

tcn_curve.eval(f"{DS_DEST}/ds/", "./weights/tcn_eidx/tcn_eidx.pth", OUTPUT_DIR, None, 1.3)
