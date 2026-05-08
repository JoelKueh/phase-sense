
import torch
import torch.nn as nn
import torchvision.models as models
from utils import utils


WINDOW_SIZE = 5
HEIGHT, WIDTH = 224, 224
SEQ_LEN = 100
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

        out = self.fc(lstm_out).squeeze(-1)  # (B, T)

        return out


# Train the LSTM
def lstm_curve_train(data_dir, weights_file):
    # Automatically determine the number of videos
    num_videos = len([f for f in os.listdir(data_dir) if f.endswith('.avi')])
    indices = list(range(num_videos))

    train_transforms = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ])
    
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_dataset = FullVideo(train_idx, SEQ_LEN, WINDOW_SIZE, None, data_dir)
    train_dataset = FullVideo(test_idx, SEQ_LEN, WINDOW_SIZE, None, data_dir)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

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

    # Save the weights to a file for testing later.
    torch.save(model.state_dict(), weights_file)
    

# Evaluate the LSTM
def lstm_curve_eval(data_dir, weights_file, output_dir, video_count):
    model = CNN_LSTM().to(device)
    model.load_state_dict(torch.load(weights_file, weights_only=True))
    model.eval()

    f = open(results_file, "w")
    for i in range(video_count):
        file_path = f"{data_dir}/{i}"
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
        plt.savefig(f"{output_dir}/figures/fig{i}.png")
        plt.clf()
