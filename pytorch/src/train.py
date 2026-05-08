import tcn_curve
import zipfile
import os

DS_TRAIN = "./datasets/ds-train.zip"
DS_DEST = "/tmp"

os.makedirs(DS_DEST, exist_ok=True)
with zipfile.ZipFile(DS_TRAIN, 'r') as zip_ref:
    zip_ref.extractall(DS_DEST)

tcn_curve.train(f"{DS_DEST}/ds/", "./weights/tcn_eidx.pth", None)
