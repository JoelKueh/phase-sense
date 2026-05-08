import tcn_curve
import zipfile
from utils.utils import *
import os

DS_TRAIN = "./datasets/ds-train.zip"
DS_DEST = "/tmp"
WEIGHTS_DEST = "./weights/tcn_onset"

os.makedirs(DS_DEST, exist_ok=True)
os.makedirs(WEIGHTS_DEST, exist_ok=True)
with zipfile.ZipFile(DS_TRAIN, 'r') as zip_ref:
    zip_ref.extractall(DS_DEST)

tcn_curve.train(f"{DS_DEST}/ds/", WEIGHTS_DEST, dtwin_onset_post_processing)
