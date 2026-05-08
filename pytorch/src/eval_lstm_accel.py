import lstm_single
import zipfile
from utils.utils import *
import os

DS_TRAIN = "./datasets/ds-eval.zip"
DS_DEST = "/tmp"
OUTPUT_DIR = "./output/lstm_accel"

os.makedirs(DS_DEST, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/curves", exist_ok=True)
with zipfile.ZipFile(DS_TRAIN, 'r') as zip_ref:
    zip_ref.extractall(DS_DEST)

WEIGHTS_PATH = "./weights/lstm_accel/lstm_accel.pth"
lstm_single.eval(f"{DS_DEST}/ds/", WEIGHTS_PATH, OUTPUT_DIR, "accel")
