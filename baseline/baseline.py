import csv

import cv2 as cv
import numpy as np
import sys
import matplotlib.pyplot as plot
from pathlib import Path

OUTDIR = "baseline_output"
# Load curve from .meta file
def load_meta(meta_path):
    data = np.loadtxt(meta_path, delimiter=',', skiprows=1)
    if data.ndim > 1:
        curve = data[:, 1]
    else:
        curve = data
    return curve

# Find the first index where the curve exceeds 1.0 (onset point)
def find_meta_onset(arr):
    for i in range(len(arr)):
        if arr[i] > 1.0:
            return i
    return 0

# Helper function to normalize an array to the range [0, 1]
def minmax(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)

# Helper to calculate cumulative error between predicted and actual curves
def cumulative_error(pred, act):
    sum = 0
    for i in range(len(pred)):
        sum += pred[i] - act[i]
    return sum

# Process a single video frame: grayscale -> Gaussian Blur -> Canny edge detection -> Morphological Closing
def frame_processing(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (21, 21), 1.0)
    edges = cv.Canny(gray, 40, 100)
    closing = cv.morphologyEx(edges, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1) #CHANGE1a: Not dilating and using 1 closing

    return closing

# Detect clusters in the processed frame and return the average cluster size in pixels.
def cluster_detection(processed_frame):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(processed_frame, connectivity=8) #Checks if pixels are connected in 8 directions (diagonal included)
    cluster_sizes = []
    output = cv.cvtColor(processed_frame, cv.COLOR_GRAY2BGR)
    for i in range(1, num_labels): # skip background label 0
        size = stats[i, cv.CC_STAT_AREA]
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        cluster_sizes.append((i, size))
        cv.rectangle(output, (x,y), (x+w, y+h), (0,255,0), 1) # DEBUGGING: Draw bounding box around cluster
    if(len(cluster_sizes) > 0):
        avg_cluster_size = np.mean([size for _, size in cluster_sizes])
    else:
        avg_cluster_size = 0
    
    # DEBUGGING: Show clusters and video feed
    cv.imshow("Clusters", output)
    cv.imshow("Video", frame)

    return avg_cluster_size

# Outputs results to output directory, returns cumulative error and onset error for csv output
def output_results(avg_array, num_frames, onset, filename, meta_path): # Output plot of predicted vs actual, abs error plot, sum of abs errors returned
    Path(f"{OUTDIR}/{filename}/").mkdir(parents=True, exist_ok=True)
    meta_curve = load_meta(meta_path)
    meta_onset = find_meta_onset(meta_curve)
    norm_act = minmax(meta_curve)
    norm_pred = minmax(avg_array)

    # Predicted vs Actual Plot
    plot.plot(range(0,len(norm_act)),norm_act, color = "blue", label = "actual")
    plot.plot(range(0, num_frames), norm_pred, color='red', linewidth=2, label = "predicted")
    plot.plot(onset, norm_pred[onset], 'o', color = "green")
    plot.xlabel("Frame")
    plot.ylabel("Emergence Index")
    plot.title("Predicted vs Actual Emergence Indicies Over Time")
    plot.legend()
    plot.savefig(f'{OUTDIR}/{filename}/{filename}_predvact.png')
    plot.clf()
    cum_error = cumulative_error(norm_pred, norm_act)
    abs_error = [abs(norm_pred[i] - norm_act[i]) for i in range(len(norm_pred))]
    onset_error = onset - meta_onset

    # Absolute Error Plot
    plot.plot(range(0,len(norm_act)),abs_error, color = "red", label = "absolute error")
    plot.xlabel("Frame")
    plot.ylabel("Absolute Error")
    plot.title("Absolute Error in Emergence Index Prediction")
    plot.legend()
    plot.savefig(f'{OUTDIR}/{filename}/{filename}_abserror.png')
    plot.clf()

    return cum_error, onset_error




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python baseline.py <video_path>")
        sys.exit(1)
    else: # Filepath parsing
        video_path = sys.argv[1]
        path_parts = video_path.split('/')
        filename_parts = path_parts[len(path_parts) - 1].split(".")
        filename = filename_parts[0]
        meta_path = f"{(video_path.split('.'))[0]}.meta"


    cap = cv.VideoCapture(video_path) # Open video file
    if not cap.isOpened():
        sys.exit(1)
    paused = False
    averages = []
    graph_data = []
    start_level = 0
    onset_time = 0
    frame_count = 0
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            result = frame_processing(frame)
            avg_cluster_size = cluster_detection(result) # In pixels
            averages.append(avg_cluster_size)
            if (start_level != 0 and onset_time == 0): # Set onset time once only
                if (avg_cluster_size >= (1.15 * start_level)): # Onset detected when average cluster size exceeds 115% of start level
                    onset_time = frame_count
                    print(f"Onset started at: {frame_count}")
            if (frame_count == 10): # Set start level as the average of the first 10 frames
                start_level = np.mean(averages[0:9])
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    for i in range(0, 13):
        graph_data.append(np.mean(averages[0:13 + i]))
    for i in range (13, len(averages) - 12):
        graph_data.append(np.mean(averages[i-13:i+13]))
    for i in range(len(averages) - 12, len(averages)):
        graph_data.append(np.mean(averages[i-13:len(averages)]))

    # Print single curve to a csv file for debugging
    # for i in range(0,frame_count):
    #     print(f"{i},{graph_data[i]}", file=open(f"debug/{filename}.csv", "a"))

    cumulative_error_sum, onset_error = output_results(graph_data, frame_count, onset_time, filename, meta_path)
    rows = {}
    path = Path(f"{OUTDIR}/results.csv")
    if path.exists():
        with path.open("r", newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    rows[row[0]] = row

    rows[filename] = [filename, cumulative_error_sum, onset_error]

    with path.open("w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows.values())


    cap.release()
    cv.destroyAllWindows()