
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from utils.utils import *


WINDOW_SIZE = 5
HEIGHT, WIDTH = 224, 224
SEQ_LEN = 100
EXTENSION = '.avi'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN_TCN(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # 1D Conv replaces LSTM
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


# Train the TCN
def train(data_dir, weights_dir, curve_post_processing):
    # Automatically determine the number of videos
    num_videos = len([f for f in os.listdir(data_dir) if f.endswith(EXTENSION)])
    indices = list(range(num_videos))

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = SlidingWindow(train_idx, SEQ_LEN, WIDTH, HEIGHT, WINDOW_SIZE, None, data_dir, curve_post_processing)
    val_dataset = SlidingWindow(test_idx, SEQ_LEN, WIDTH, HEIGHT, WINDOW_SIZE, None, data_dir, curve_post_processing)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Train the model
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item()
        torch.save(model.state_dict(), f"{weights_dir}/{epoch}_{val_loss}.pth")
        print(f"Validation Loss: {val_loss}")


# Helper function to predict a single video curve.
def predict_video_curve(model, window_size, video_path):
    model.eval()
    frames = frames_extraction(video_path, SEQ_LEN, WIDTH, HEIGHT)
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


# Evaluate the TCN
def eval(data_dir, weights_file, output_dir, curve_post_processing, act_thresh, pred_thresh):
    # Automatically determine the number of videos
    num_videos = len([f for f in os.listdir(data_dir) if f.endswith(EXTENSION)])
    indices = list(range(num_videos))

    model = CNN_TCN().to(device)
    model.load_state_dict(torch.load(weights_file, weights_only=True))
    model.eval()

    f = open(f"{output_dir}/results.csv", "w")
    for i in range(num_videos):
        file_path = f"{data_dir}/{i}"
        print(f"Analyzing Video: {i}")
        curve = load_meta(f"{file_path}.meta", SEQ_LEN)
        if curve_post_processing:
            curve = curve_post_processing(curve)
        curve = curve.numpy()
        plt.plot(curve, label="True")
        pred = predict_video_curve(model, WINDOW_SIZE, f"{file_path}.avi")
        print(f"{i},\
              {3 * two_way_minmax_curve_error(pred, curve)},\
              {3 * onset_error(pred, curve[4:], act_thresh, pred_thresh)}",\
              file=f, flush=True)
        plt.plot(range(4,4+len(pred)), pred, label="Pred")
        plt.legend()
        plt.title("Emergence Curve Prediction [0]")
        plt.savefig(f"{output_dir}/figures/fig{i}.png")
        with open(f"{output_dir}/curves/{i}.csv", "w") as curvef:
            for i, val in enumerate(pred):
                print(f"{3*(i + WINDOW_SIZE - 1)},{val}", file=curvef)
        plt.clf()
