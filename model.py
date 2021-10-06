from cv2 import VideoCapture

from torch import device
from torchvision.models.detection import keypointrcnn_resnet50_fpn as RCNNModel

from utils import img2points, centerCrop
import time


def getPoints(queue):

    #initialize model
    KeypointRCNN = RCNNModel(pretrained=True, progress=True)
    KeypointRCNN.cuda().eval()
    KeypointRCNN.to(device('cuda:0'))

    #set up CV2 video capture
    cap = VideoCapture(0)
    cap.set(3, 1920)
    cap.set(4, 1080)

    while True:

        # run inference
        _, img = cap.read()
        img = centerCrop(img, 1000)

        point_list = img2points(img, KeypointRCNN)
        
        # add the most recent pt list
        queue.put((img, point_list))

        # if the previous pt list has not been taken by main, discard it
        if queue.qsize() > 1:
            _ = queue.get()


def getPointsAsync(queue):
    import cv2

    #initialize model
    KeypointRCNN = RCNNModel(pretrained=True, progress=True)
    KeypointRCNN.cuda().eval()
    KeypointRCNN.to(device('cuda:0'))

    #set up CV2 video capture
    cap = VideoCapture(r"I:\TRS Project 2\BodyBrush\input_video.mp4")

    while True:
        print("frame grabbing is running")

        # run inference
        flag, img = cap.read()
        print(flag, img.shape)
        
        if flag:
            img = centerCrop(img, 1000)

            point_list = img2points(img, KeypointRCNN)

            time.sleep(3)
        
            # add the most recent pt list
            queue.put((img, point_list))

            # if the previous pt list has not been taken by main, discard it
            if queue.qsize() > 1:
                _ = queue.get()

            print("frame inference complete")

        else:
                # The next frame is not ready, so we try to read it again
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)