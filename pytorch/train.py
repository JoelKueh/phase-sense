
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


class VideoDataset(Dataset):
    def __init__(self, indices, window_size=5, transform=None):
        self.transform = transform
        self.window_size = window_size
        self.len = len(indices)
        self.data = []

        for i in tqdm(range(len(indices))):
            idx = indices[i]
            video_path = os.path.join(DATA_DIR, f"{idx}.avi")
            meta_path = os.path.join(DATA_DIR, f"{idx}.meta")

            frames = frames_extraction(video_path)
            curve = load_meta(meta_path)
            curve = resize_curve(curve)
            self.data.append((frames, curve))

    def __len__(self):
        return self.len * (SEQ_LEN - self.window_size + 1)

    def __getitem__(self, idx):
        video_idx = idx // (SEQ_LEN - self.window_size + 1)
        frame_idx = idx % (SEQ_LEN - self.window_size + 1)
        frames, curve = self.data[video_idx]
        frames = frames[frame_idx:frame_idx+self.window_size]
        target = curve[frame_idx+4]
        if self.transform:
            frames = self.transform(frames)
        return frames, target

# Automatically determine the number of videos
num_videos = len([f for f in os.listdir(DATA_DIR) if f.endswith('.avi')])
indices = list(range(num_videos))

# Define transformations for data augmentation
train_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.1, contrast=0.1),
    v2.ToDtype(torch.float32, scale=True),
])

train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_dataset = VideoDataset(train_idx, window_size=5)
test_dataset  = VideoDataset(test_idx, window_size=5)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


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


print(f"Training on device: {device}")
model = CNN_TCN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 10

for epoch in range(EPOCHS):
    print(f"Beginning Epoch {epoch}")
    model.train()
    train_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = model(xb)
        loss = criterion(preds, yb)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

torch.save(model.state_dict(), "./model_weights.pth")
model.eval()
test_loss = 0

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)

        preds = model(xb)
        loss = criterion(preds, yb)

        test_loss += loss.item()

print("Test Loss:", test_loss)
