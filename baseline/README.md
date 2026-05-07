
# Computer Vision Baseline

This is the computer vision component of the PhaseSense project.

## Dependencies

This project has 1 main dependency:
1. OpenCV - Open Source Computer Vision Library
  -  Install using pip install opencv-python
  -  This is an open source library that has all the computer vision related methods that were needed for this project.

## Design Philosophy

This repository implements a computer vision baseline that uses a combination of computer vision techniques to detect clusters for the formation of the emergence curve and prediction of the onset time.

### Cluster Detection

The pipeline uses OpenCV as its main platform and facilitator. It takes in the source video and processes it frame by frame using the function: 
```python
def frame_processing(frame):
```
For each frame in the video, the following operations are performed:
1. Initial grayscale and Gaussian blur filter applied.
2. Canny edge detection with bounds of (40,100).
3. A morphological closing operation with a small 3x3 kernel and just 1 iteration.

After this, the result is passed into the 
```py 
def cluster_detection(processed_frame)
```
method. A single cluster is defined as a connected cluster of white pixels. This cluster detection method uses the ```cv.connectedComponentsWithStats()``` method to obtain the area of each cluster. Then, the average of all the cluster sizes is taken and returned. This single average is what is used as a single emergence index. This serves as an estimate of the effect of emergence index that is produced by the digital twin pipeline and trained on by the machine learning pipeline. However, for the two indexing systems to be comparable, they both are later normalized to a 0-1 range system.

### Emergence Curve Formation
A compilation of all the emergence indices over all frames forms what is known as the emergence curve. This curve displays the effect of the onset and clustering of the particles with a sharp upturn at this event. As the time goes on, the curve levels off as the phenomenon stabilizes.

The emergence curve for this program is formed using a moving average approach. For the ends of the curve which are the first and last 13 values, the average is taken over a window growing from 13 to 26 values in the beginning and shrinking from 26 to 13 values at the end. This is to have the beginning and end frames not initially covered by the traditional sliding window included. In the middle of these ranges, a sliding window of 26 frames is used. This approach is mainly to stabilize and smoothen the curve to display the desired effect of the particle aggregation. The end result is a curve that generally follows a sigmoid shape.

### Onset Time Estimate

To estimate the onset time in real time as the video is being processed, a thresholding approach was implemented. A separate average of the first 10 frames is computed and used as a baseline threshold; onset is detected when the emergence index rises above this threshold by 15%. This number was chosen due to the accuracy it provided in the case of simulated asbestos particles. Once the onset is predicted, a warning message is printed to the terminal and the frame number is recorded and later marked on the curve.

## Running

### Input and Usage

To run the baseline.py program:
```sh
python path/to/baseline.py path/to/video.avi
```
Note: mp4, avi, and any other video codecs supported by OpenCV are supported here. Instead of python, some systems may require python3 instead.

To run the test.py test script:
```sh
python path/to/test.py path/to/video_directory
```
Note: The video directory is expected to be exactly half video files and half corresponding .meta files. The test script runs the program once for each video file.
### Output

The pipeline will output a predicted vs. actual emergence curve and an absolute error curve to `./OUTDIR/videoname`. The predicted data referenced is the computer vision baseline data, while the actual data produced by the digital twin is pulled from the video's corresponding `.meta` file.

The pipeline also outputs real-time onset detection to the terminal as discussed above.

A line is also appended to the results.csv file (if not already existing) in the form ```(video name, cumulative curve error, onset prediction error)```.

Some optional debugging outputs are the original video feed (can be slowed with debugging sleep parameter included), cluster detection display, graph visualization on demand before being saved to the corresponding directory, and single curve to csv.