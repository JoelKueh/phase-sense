
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

from utils.utils import *


HEIGHT, WIDTH = 224, 224
SEQ_LEN = 100
EXTENSION = '.avi'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        lstm_out = lstm_out[:, -1, :]

        out = self.fc(lstm_out).squeeze(-1)  # (B, T)

        return out


# Train the LSTM
def train(data_dir, weights_dir, param_name):
    # Automatically determine the number of videos
    num_videos = len([f for f in os.listdir(data_dir) if f.endswith('.avi')])
    indices = list(range(num_videos))

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = FullVideoSingle(train_idx, SEQ_LEN, WIDTH, HEIGHT, None, data_dir, param_name)
    val_dataset = FullVideoSingle(test_idx, SEQ_LEN, WIDTH, HEIGHT, None, data_dir, param_name)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # Train the model
    model = CNN_LSTM().to(device)
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
    

def predict_param(model, video_path):
    model.eval()
    frames = frames_extraction(video_path, SEQ_LEN, WIDTH, HEIGHT)
    if frames is None:
        return None

    predictions = []
    with torch.no_grad():
        input_tensor = frames.unsqueeze(0).to(device)
        pred = model(input_tensor)
        predictions.append(pred.item())
    return np.array(predictions)
    

# Evaluate the LSTM
def eval(data_dir, weights_file, output_dir, param_name):
    # Automatically determine the number of videos
    num_videos = len([f for f in os.listdir(data_dir) if f.endswith(EXTENSION)])
    indices = list(range(num_videos))

    model = CNN_LSTM().to(device)
    model.load_state_dict(torch.load(weights_file, weights_only=True))
    model.eval()

    f = open(f"{output_dir}/results.csv", "w")
    for i in range(num_videos):
        file_path = f"{data_dir}/{i}"
        print(f"Analyzing Video: {i}")
        target = meta_get_param(f"{file_path}.meta", param_name)
        target = target.numpy()
        pred = predict_param(model, f"{file_path}.avi")
        print(f"{i},{target},{pred}",file=f, flush=True)
    close(f)
