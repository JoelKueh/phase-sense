
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
import os
import cv2


## Video Loading Utilities

# Extracts the frames from a given video path
def frames_extraction(video_path, seq_len, width, height):
    frames = []
    cam = cv2.VideoCapture(video_path)
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, frame_count-1, seq_len, dtype=int)

    for idx in indices:
        cam.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))
        frame = frame / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        frames.append(torch.from_numpy(frame.astype(np.float32)))

    cam.release()
    return torch.stack(frames) if len(frames) > 0 else None


# Loads the metadata for a video at the given path
def load_meta(meta_path, seq_len):
    data = np.loadtxt(meta_path, delimiter=',', skiprows=1)
    if data.ndim > 1:
        curve = data[:, 1]
    else:
        curve = data

    curve = torch.tensor(curve, dtype=torch.float32)
    curve = curve.unsqueeze(0).unsqueeze(0)
    curve = F.interpolate(curve, size=seq_len, mode='linear', align_corners=False)
    return curve.squeeze()


# Turns an emergence index sequence into an onset chain
def dtwin_onset_post_processing(curve):
    new = [1.0 if x > 1.00001 else 0.0 for x in curve]
    return torch.tensor([np.float32(x) for x in new])


## Dataset Utilitiies

class SlidingWindow(Dataset):
    def __init__(self, indices, seq_len, width, height, window_size, transform,
                 data_dir, curve_post_processing):
        self.transform = transform
        self.window_size = window_size
        self.seq_len = seq_len
        self.width = width
        self.height = height
        self.data_dir = data_dir
        self.len = len(indices)
        self.data = []

        for i in tqdm(range(len(indices))):
            idx = indices[i]
            video_path = os.path.join(self.data_dir, f"{idx}.avi")
            meta_path = os.path.join(self.data_dir, f"{idx}.meta")

            frames = frames_extraction(video_path, self.seq_len, self.width, self.height)
            curve = load_meta(meta_path, self.seq_len)
            if curve_post_processing:
                curve = curve_post_processing(curve)
            self.data.append((frames, curve))

    def __len__(self):
        return self.len * (self.seq_len - self.window_size + 1)

    def __getitem__(self, idx):
        video_idx = idx // (self.seq_len - self.window_size + 1)
        frame_idx = idx % (self.seq_len - self.window_size + 1)
        frames, curve = self.data[video_idx]
        frames = frames[frame_idx:frame_idx+self.window_size]
        target = curve[frame_idx+4]
        if self.transform:
            frames = self.transform(frames)
        return frames, target


class FullVideo(Dataset):
    def __init__(self, indices, seq_len, width, height, transform,
                 data_dir, curve_post_processing):
        self.transform = transform
        self.seq_len = seq_len
        self.width = width
        self.height = height
        self.data_dir = data_dir
        self.len = len(indices)
        self.data = []

        for i in tqdm(range(len(indices))):
            idx = indices[i]
            video_path = os.path.join(self.data_dir, f"{idx}.avi")
            meta_path = os.path.join(self.data_dir, f"{idx}.meta")

            frames = frames_extraction(video_path, self.seq_len, self.width, self.height)
            curve = load_meta(meta_path, self.seq_len)
            if curve_post_processing:
                curve = curve_post_processing(curve)
            self.data.append((frames, curve))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        frames, curve = self.data[idx]
        if self.transform:
            frames = self.transform(frames)
        return frames, curve


## Data Comparison Utilities

# Normalizes a curve onto the range [0,1]
def minmax(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if (arr_min == arr_max):
        return arr / np.average(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def avgnormal(arr):
    return arr / np.average(arr)


# Normalizes pred and act and compares them
def two_way_minmax_curve_error(pred, act):
    sum = 0
    if np.min(act) == np.max(act):
        norm_pred = avgnormal(pred)
        norm_act = avgnormal(act)
    else:
        norm_pred = minmax(pred)
        norm_act = minmax(act)
    for i in range(len(pred)):
        sum += norm_pred[i] - norm_act[i]
    return sum


# Finds the onset in the provided curve via thresholding.
def find_onset(curve, thresh):
    for i in range(len(curve)):
        if curve[i] > thresh:
            return i
    return len(curve)


# Computes the error in the onset prediction from the actual
def onset_error(pred, act, act_thresh, pred_thresh):
    act_onset = find_onset(act, act_thresh)
    pred_onset = find_onset(pred, pred_thresh)
    return (pred_onset - act_onset)


