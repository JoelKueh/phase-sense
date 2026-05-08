import lstm_single
import zipfile
import os

DS_TRAIN = "./datasets/ds-train.zip"
DS_DEST = "/tmp"
WEIGHTS_DEST = "./weights/lstm_accel"

os.makedirs(DS_DEST, exist_ok=True)
os.makedirs(WEIGHTS_DEST, exist_ok=True)
with zipfile.ZipFile(DS_TRAIN, 'r') as zip_ref:
    zip_ref.extractall(DS_DEST)

lstm_single.train(f"{DS_DEST}/ds/", WEIGHTS_DEST, "accel")
