
import torch
import torch.nn as nn
import torchvision.models as models
from utils import utils


WINDOW_SIZE = 5
HEIGHT, WIDTH = 224, 224
SEQ_LEN = 100
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
def tcn_onset_train(data_dir, weights_file):
    # Automatically determine the number of videos
    num_videos = len([f for f in os.listdir(data_dir) if f.endswith('.avi')])
    indices = list(range(num_videos))

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = SlidingWindow(train_idx, SEQ_LEN, WINDOW_SIZE, None, data_dir)
    train_dataset = SlidingWindow(test_idx, SEQ_LEN, WINDOW_SIZE, None, data_dir)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

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

    # Save the weights to a file for testing later.
    torch.save(model.state_dict(), weights_file)
    

# Helper function to predict a single video curve.
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


# Evaluate the TCN
def tcn_onset_eval(data_dir, weights_file, output_dir, video_count):
    model = CNN_TCN().to(device)
    model.load_state_dict(torch.load(weights_file, weights_only=True))
    model.eval()

    f = open(results_file, "w")
    for i in range(video_count):
        file_path = f"{data_dir}/{i}"
        print(f"Analyzing Video: {i}")
        curve = load_meta(f"{file_path}.meta")
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
        plt.savefig(f"{output_dir}/figures/fig{i}.png")
        plt.clf()
