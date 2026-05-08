import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile


DS_TRAIN = "./pytorch/datasets/ds-eval.zip"
DS_DEST = "/tmp"
OUTPUT_DIR = "./plots"
os.makedirs(DS_DEST, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
with zipfile.ZipFile(DS_TRAIN, 'r') as zip_ref:
    zip_ref.extractall(DS_DEST)


DATA_ML = './pytorch/output/tcn_eidx/results.csv'
DATA_ONSET_ML = './pytorch/output/tcn_onset/results.csv'
DATA_BASE = './baseline/baseline_output/results.csv'
CURVE_ACTUAL = f'{DS_DEST}/ds/31.meta'
CURVE_ML = './pytorch/output/tcn_eidx/curves/31.csv'
CURVE_BASE = './baseline/baseline_output/31/curve.csv'


def minmax(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def avgnormal(arr):
    return arr / np.average(arr)


data_ml = np.loadtxt(DATA_ML, delimiter=',', skiprows=0)
data_ml = np.transpose(data_ml)
data_onset_ml = np.loadtxt(DATA_ONSET_ML, delimiter=',', skiprows=0)
data_onset_ml = np.transpose(data_onset_ml)
data_base = np.loadtxt(DATA_BASE, delimiter=',', skiprows=0)
data_base = np.transpose(data_base)

curve_actual = np.transpose(np.loadtxt(CURVE_ACTUAL, delimiter=',', skiprows=1))
curve_ml = np.transpose(np.loadtxt(CURVE_ML, delimiter=',', skiprows=0))
curve_cv = np.loadtxt(CURVE_BASE, delimiter=',', skiprows=0)
curve_cv = np.transpose(curve_cv)

scale = 0.35
fs = [24 * scale, 11 * scale]
fig,ax = plt.subplots(figsize=fs)
ax.set_title("Normalized Emergence Curve Comparison")
ax.plot(curve_actual[0], minmax(curve_actual[1]), label='Actual')
ax.plot(curve_ml[0], minmax(curve_ml[1]), label='PyTorch')
ax.plot(curve_cv[0], minmax(curve_cv[1]), label='Computer Vision')
ax.legend()
ax.set_xlabel("Frame Number")
ax.set_ylabel("Normalized Emergence Index")
plt.savefig(f"{OUTPUT_DIR}/result_norm.png", pad_inches=0)
plt.clf()

w=4.0
fig,ax = plt.subplots(figsize=fs)
ax.set_title("Onset Estimate Error")
counts, bins = np.histogram(data_onset_ml[2], bins=50)
ax.bar(bins[:-1]-w/2, counts, width=w, label='PyTorch')
counts, _ = np.histogram(data_base[2], bins=bins)
ax.bar(bins[:-1]+w/2, counts, width=w, label='Computer Vision')
ax.set_xlabel("Number of Frames")
ax.legend()
plt.savefig(f"{OUTPUT_DIR}/result_onset_error.png", pad_inches=0)
plt.clf()

mask_ml = data_ml[2] < 15
filt_ml = data_ml[2][mask_ml]
mask_base = data_base[2] < 15
filt_base = data_base[2][mask_base]
print("Average ML Onset Error: ", np.average(np.abs(data_ml[2])))
print("Average Base Onset Error: ", np.average(np.abs(data_base[2])))
print("ML Prediction Accuracy: ", len(filt_ml) / len(data_ml[2]))
print("Base Prediction Accuracy: ", len(filt_base) / len(data_base[2]))

w=1.0
fig,ax = plt.subplots(figsize=fs)
ax.set_title("Cumulative Error in Normalized Emergence Curves")
counts, bins = np.histogram(data_ml[1], bins=50)
ax.bar(bins[:-1]-w/2, counts, width=w, label='PyTorch')
counts, _ = np.histogram(data_base[1], bins=bins);
ax.bar(bins[:-1]+w/2, counts, width=w, label='Computer Vision')
ax.legend()
plt.savefig(f"{OUTPUT_DIR}/result_cumulative_error.png", pad_inches=0)
plt.clf()
print("Average ML Cumulative Error: ", np.average(np.abs(data_ml[1])))
print("Average Base Cumulative Error: ", np.average(np.abs(data_base[1])))

