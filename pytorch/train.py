
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
    curve = (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)

    return curve


def resize_curve(curve, target_len=SEQ_LEN):
    curve = curve.unsqueeze(0).unsqueeze(0)  # (1,1,T)
    curve = F.interpolate(curve, size=target_len, mode='linear', align_corners=False) #resizes curve to targe_len
    return curve.squeeze()


class VideoDataset(Dataset):
    def __init__(self, indices, transform=None):
        self.transform = transform
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
        return self.len

    def __getitem__(self, idx):
        frames, curve = self.data[idx]
        if self.transform:
            frames = self.transform(frames)
        return frames, curve

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
train_dataset = VideoDataset(train_idx, transform=train_transforms) # Apply transforms to training data
test_dataset  = VideoDataset(test_idx) # No augmentation for test set

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

# Select a single video from the test set for demonstration
single_video_idx = test_idx[0]
single_video_path = os.path.join(DATA_DIR, f"{single_video_idx}.mp4")

# Extract frames using the modified function
extracted_frames = frames_extraction(single_video_path)

if extracted_frames is not None:
    print(f"Extracted frames shape: {extracted_frames.shape}")

    # Visualize the extracted frames
    fig, axes = plt.subplots(1, SEQ_LEN, figsize=(30, 4))
    for i in range(SEQ_LEN):
        # Convert C, H, W back to H, W, C for plotting with matplotlib
        img_to_plot = extracted_frames[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img_to_plot)
        axes[i].axis('off')
    plt.suptitle(f"Extracted Frames from Video {single_video_idx}", fontsize=16)
    plt.tight_layout()
    plt.show()
else:
    print(f"Could not extract frames from video: {single_video_path}")


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
model = CNN_LSTM().to(device)
criterion = nn.HuberLoss(delta=0.1)
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

model.eval()

xb, yb = next(iter(test_loader))
xb = xb.to(device)

with torch.no_grad():
    preds = model(xb).cpu()

yb = yb.cpu()

print("Plotting first emergence curve set:")
plt.plot(yb[0], label="True")
plt.plot(preds[0], label="Pred")
plt.legend()
plt.title("Emergence Curve Prediction [0]")
plt.savefig("fig1.png")

print("Plotting second emergence curve set:")
plt.figure()
plt.plot(yb[1], label="True")
plt.plot(preds[1], label="Pred")
plt.legend()
plt.title("Emergence Curve Prediction [1]")
plt.savefig("fig2.png")

# 2. Put the model in evaluation mode
model.eval()

# Set up the plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Emergence Curve Predictions for Real-Life Videos", fontsize=16)

with torch.no_grad(): # No need to track gradients for inference
    for i, video_path in enumerate(real_video_paths):
        if not os.path.exists(video_path):
            print(f"File not found: {video_path}. Please check the path.")
            continue

        print(f"Processing {video_path}...")

        # 3. Extract and preprocess frames
        frames_tensor = frames_extraction(video_path)

        if frames_tensor is None:
            print(f"Failed to extract frames from {video_path}")
            continue

        # 4. Add batch dimension: (T, C, H, W) -> (1, T, C, H, W)
        frames_batch = frames_tensor.unsqueeze(0).to(device)

        # 5. Get the prediction
        predicted_curve = model(frames_batch)

        # Move back to CPU and remove batch dimension for plotting
        predicted_curve = predicted_curve.cpu().squeeze().numpy()

        # 6. Plot the predicted curve
        axes[i].plot(predicted_curve, label="Predicted Emergence", color='red', linewidth=2)
        axes[i].set_title(f"Video {i+1}")
        axes[i].set_xlabel("Time (Sequence Length)")
        axes[i].set_ylabel("Normalized Cluster Size")
        axes[i].set_ylim(0, 1.1) # Assuming your curves are normalized 0-1
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend()

plt.tight_layout()
plt.show()

