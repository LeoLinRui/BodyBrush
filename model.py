import numpy as np
import cv2

import torch
import torchvision

from utils import img2points

device = 'cuda:0'


def getPoints(queue):

    #initialize model
    KeypointRCNN = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, progress=True)
    KeypointRCNN.cuda().eval()
    KeypointRCNN.to(torch.device(device))

    #set up CV2 video capture
    cap = cv2.VideoCapture(0)
    cap.set(3, 1080)
    cap.set(4, 1080)

    while True:

        # run inference
        _, img = cap.read()
        point_list = img2points(img, KeypointRCNN)
        
        # add the most recent pt list
        queue.put((img, point_list))

        # if the previous pt list has not been taken by main, discard it
        if queue.qsize() > 1:
            _ = queue.get()