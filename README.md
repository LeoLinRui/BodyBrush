# BodyBrush
#### For TRS Project #2

## Program Structure
- main.py: main script for synchronous processing. Initialize all sub-processes: 1 inference subprocess and 8 renderer worker processes. 3 queues are used for inter-process communication.
- async.py: a copy of main designed for asynchronous processing of a given video footage.
- model.py: inference module. Captures video from webcam, run inference on Keypoint RCNN, and put result into a queue while maintaining the queue. An async version of the function is also available in the file.
- modules.py: contains all the class definitions for: the Path class, the PathManager class, and the renderer method.
- utils.py: contains the helper functions used for inference, rendering, processing, debugging, etc.

## Digital Brush Design
The brush strokes are managed by the Path class. The Path class is designed to suppoert brush shape, brush grain, jitter, smoothing, variable scale, color, vector calculation for shape rotations, temporal decay, etc. The data is stored as a vector-like path that can be rendered with different resolutions. Note that not all of the functions are currently implemented.

The following sampe of a brush stroke demonstrate the capability of what has been implemented:
![output](https://user-images.githubusercontent.com/44302577/136150395-771a3852-a415-4417-94c0-d288e1bdcac4.png)
