
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt

import cv2
import os
import zipfile

real_video_paths = [
    "/content/Real-Time-MicroVids/aggregation_of_silica_spheres.avi", # Change these to your actual file names
    "/content/Real-Time-MicroVids/aggregation_of_silica_rods.avi",
    "/content/Real-Time-MicroVids/aggregation_of_chrysotile_fibers.avi"
]

FRAME_COUNT = 300
HEIGHT, WIDTH = 224, 224
SEQ_LEN = 100
DATA_DIR = "../dtwin/out/ds/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"ROCm avaliable? {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
print(f"Device: {device}")


def frames_extraction(video_path):
    frames = []
    cam = cv2.VideoCapture(video_path)
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    assert(frame_count == FRAME_COUNT)
    indices = np.linspace(0, frame_count-1, SEQ_LEN, dtype=int)

    for idx in indices:
        cam.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(torch.from_numpy(frame.astype(np.float32)))

    cam.release()
    return torch.stack(frames) if len(frames) > 0 else None


def load_meta(meta_path):
    data = np.loadtxt(meta_path, delimiter=',', skiprows=1)
    if data.ndim > 1:
        curve = data[:, 1]
    else:
        curve = data

    curve = torch.tensor(curve, dtype=torch.float32)
    return curve


def resize_curve(curve, target_len=SEQ_LEN):
    curve = curve.unsqueeze(0).unsqueeze(0)  # (1,1,T)
    curve = F.interpolate(curve, size=target_len, mode='linear', align_corners=False) #resizes curve to targe_len
    return curve.squeeze()


class CNN_TCN(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # 1D Conv replaces LSTM
        # padding=window_size // 2 keeps the sequence length at 100
        self.temporal_filter = nn.Conv1d(in_channels=2048, 
                                         out_channels=64, 
                                         kernel_size=window_size, 
                                         padding=window_size // 2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        # 1. Spatial: Process frames in ResNet
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x) # (B*T, 2048, 1, 1)
        feats = self.pooling(feats).view(B, T, 2048)

        # 2. Temporal: Process windows in a TCN
        feats = feats.permute(0, 2, 1)
        temporal_feats = F.relu(self.temporal_filter(feats))

        # 3. Collapse Time: Average 5 frames into 1 vector
        combined_features = torch.mean(temporal_feats, dim=2)
        return self.fc(combined_features).squeeze(-1)


class CNN_LSTM(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()

        # instantiate resnet 50 model
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # fine-tune last block only of resnet CNN layers
        for name, param in self.cnn.named_parameters():
            param.requires_grad = "layer4" in name

        # create LSTM layer to match the output of the resnet50 model
        self.lstm = nn.LSTM(2048, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)
        feats = self.cnn(x)
        feats = self.pooling(feats)
        feats = feats.view(B, T, 2048)

        lstm_out, _ = self.lstm(feats)
        lstm_out = self.dropout(lstm_out)

        out = self.fc(lstm_out).squeeze(-1)  # (B, T)

        return out

model = CNN_TCN().to(device)
model.load_state_dict(torch.load("./model_weights.pth", weights_only=True))
model.eval()


window_size = 5
def predict_video_curve(video_path):
    model.eval()
    frames = frames_extraction(video_path)
    if frames is None:
        return None

    predictions = []
    with torch.no_grad():
        for i in range(len(frames) - window_size + 1):
            window = frames[i : i + window_size]
            input_tensor = window.unsqueeze(0).to(device)
            pred = model(input_tensor)
            predictions.append(pred.item())
    return np.array(predictions)


def cumulative_error(pred, act):
    sum = 0
    for i in range(len(pred)):
        sum += pred[i] - act[i]
    return sum


def minmax(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


# Normalizes pred and act and compares them
def two_way_minmax_curve_error(pred, act):
    sum = 0
    norm_pred = minmax(pred)
    norm_act = minmax(act)
    for i in range(len(pred)):
        sum += norm_pred[i] - norm_act[i]
    return sum


# Normalizes act and maps applies same transform on pred
def one_way_minmax_curve_error(pred, act):
    sum = 0
    arr_min = np.min(act)
    arr_max = np.max(act)
    norm_pred = (pred - arr_min) / (arr_max - arr_min)
    norm_act = (act - arr_min) / (arr_max - arr_min)
    for i in range(len(pred)):
        sum += norm_pred[i] - norm_act[i]
    return sum


def onset_error(pred, act):
    thresh = 1.01
    act_onset = 0
    for i in range(len(act)):
        if act[i] > thresh:
            act_onset = i
            break
    thresh = 1.5
    pred_onset = 0
    for i in range(len(pred)):
        if pred[i] > thresh:
            pred_onset = i+4
            break
    return 3 * (pred_onset - act_onset)


f = open("results.csv", "w")
for i in range(250):
    file_path = f"../dtwin/out/ds/{i}"
    print(f"Analyzing Video: {i}")
    curve = load_meta(f"{file_path}.meta")
    curve = resize_curve(curve)
    curve = curve.numpy()
    plt.plot(curve, label="True")
    pred = predict_video_curve(f"{file_path}.avi")
    print(f"{i},\
          {cumulative_error(pred, curve)},\
          {onset_error(pred, curve)},\
          {two_way_minmax_curve_error(pred, curve)},\
          {one_way_minmax_curve_error(pred, curve)}",\
          file=f, flush=True)
    plt.plot(range(4,4+len(pred)), pred, label="Pred")
    plt.legend()
    plt.title("Emergence Curve Prediction [0]")
    plt.savefig(f"fig{i}.png")
    plt.clf()
    


