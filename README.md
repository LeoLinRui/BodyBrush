# BodyBrush
### For TRS Project #2
Work in progress...

## Project File Structure
- **main.py:** main script for synchronous processing. Initialize all sub-processes: 1 inference subprocess and 8 renderer worker processes. 3 queues are used for inter-process communication.
- **async.py:** a copy of main designed for asynchronous processing of a given video footage.
- **model.py:** inference module. Captures video from webcam, run inference on Keypoint RCNN, and put result into a queue while maintaining the queue. An async version of the function is also available in the file.
- **modules.py:** contains all the class definitions for: the Path class, the PathManager class, and the renderer method.
- **utils.py:** contains the helper functions used for inference, rendering, processing, debugging, etc.
- **sketch.ipynb:** A jupyter notebook used for sketching.

## Digital Brush Design
The brush strokes are managed by the Path class. The Path class is designed to suppoert brush shape, brush grain, jitter, smoothing, variable scale, variable color, taper, vector calculation for shape rotations, temporal decay, etc. The data is stored as a vector-like path that can be rendered with different resolutions. Note that not all of the functions are currently implemented.

The following sampe of a brush stroke demonstrate the capability of what has been implemented:
<p align="center">
<img src="https://user-images.githubusercontent.com/44302577/136150395-771a3852-a415-4417-94c0-d288e1bdcac4.png" alt="drawing" width="600"/>
</p>

## Human Keypoint Detection
KeypointRCNN is used to detect the location of 17 keypoints, which are used to create the brushstrokes. 

A demo of KeypointRCNN can be found using the link below:

https://user-images.githubusercontent.com/44302577/136155469-a9f0d20c-4960-499e-8182-105a12763c40.mp4

## Synchronous Inference
As of now, synchronous inference and rendering will run at about 1.2 s/frame. Multiprocessing of RCNN inference, Alpha blending, and path rendering is implemented so they run in parallel. Alpha blending currently takes ~900ms and rendering takes ~1100ms. Possibly due to the multiprocessing and IPC overhead, this could not be accelerated any further.

A sample recording of real-time inference can be found using the link below:

https://user-images.githubusercontent.com/44302577/136153660-60917abf-bd55-40f5-9035-ec7517ee4595.mp4

## Asynchronous Inference
Bug as feature I guess... I couldn't figure out how to make my code run any faster but I need something presentable for the crit. Maybe it's time to learn c.

A sample os async output can also be found using the link below:

https://user-images.githubusercontent.com/44302577/136154225-0ee2b2d9-049e-4f45-959b-b344e42a6fd3.mp4

## Inspiration & Sources
Besides the various artists we viewed in class for this project, there are some other artists whose work inspired and informed this project.
Some technical reference material are also linked here.

### Hiroshi Sugimoto's Theatre
<p align="center">
<img src="https://user-images.githubusercontent.com/44302577/136246526-6ad4edf5-c1bf-4f6f-b11a-5ff2c61cf6fd.jpg" alt="drawing" width="600"/>
</p>

### The Fence
