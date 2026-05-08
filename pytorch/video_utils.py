
import torch
import cv2


def frames_extraction(video_path, seq_len):
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


def load_meta(meta_path, seq_len):
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
