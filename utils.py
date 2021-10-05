import os
import io
import time
import math
import pandas as pd
import numpy as np
import multiprocessing

import cv2
from skimage import transform
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional

device = 'cuda:0'


#====================================================================================================
# Inference and debugging helper functions
#====================================================================================================  


def to_variable(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

to_tensor = transforms.ToTensor()

def show(img): # Display rgb tensor image
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)
    
def show_gray(img): # Display grayscale tensor image
    pilTrans = transforms.ToPILImage()
    pilImg = pilTrans(img)
    s = np.array(pilImg)
    
    cv2.namedWindow( "Display window", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Display window", s)
    cv2.waitKey(0)
    
    '''
    plt.figure()
    plt.imshow(s)
    '''
    
def predict(model, img):
    im = to_tensor(img).cuda()
    im = to_variable(im.unsqueeze(0), False)
    #im = im.cuda()
    im = im.to(torch.device(device))
    out = model(im)
    return out

def show_img_from_path(imgPath):
    pilImg = Image.open(imgPath)
    size = np.array(pilImg.size)
    size = np.append(size, 3).reshape(3)
    print(size)
    s = np.array(pilImg)
    plt.figure()
    plt.imshow(s)


#====================================================================================================
# KeyPointRCNN inference function
#====================================================================================================   


def img2points(img, model):
    point_list = [] # this is the return

    predictions = predict(model, img)

    threshold = 0.93 # 0.93
    kpt_score_threshold = 5 # 5

    patches = []
    idx = 0

    idx_of_best_box = np.argmax(predictions[0]['scores'].detach().cpu().numpy())

    '''box_idx = 0
    for dot_group in predictions[0]['keypoints']: # predictions[0]['keypoints'] in FloatTensor[N, K, 3]
        point_idx = 0
        for dot in dot_group:
            if predictions[0]['keypoints_scores'][box_idx][point_idx] >= kpt_score_threshold:
                if predictions[0]['scores'][box_idx] >= threshold:
                    point_list.append([point_idx, dot[0].cpu().detach().numpy(), dot[1].cpu().detach().numpy()])
            point_idx += 1
        box_idx += 1'''

    box_idx = 0
    for dot_group in predictions[0]['keypoints']: # predictions[0]['keypoints'] in FloatTensor[N, K, 3]
        point_idx = 0
        for dot in dot_group:
            if predictions[0]['keypoints_scores'][box_idx][point_idx] >= kpt_score_threshold:
                if predictions[0]['scores'][box_idx] >= threshold and box_idx == idx_of_best_box:
                    point_list.append([point_idx, dot[0].cpu().detach().numpy(), dot[1].cpu().detach().numpy()])
            point_idx += 1
        box_idx += 1
    

    return point_list


#====================================================================================================
# renderer process helper functions
#====================================================================================================  


def alphaBlend(top, bottom):
    '''
    takes an HSLA top and a bottom, blends and returns a HSL image
    '''
    assert top.shape[0] == 4, "top must have alpha channel"
    assert top.shape[1] == bottom.shape[1], "top and bottom must have equal shape"
    assert top.shape[2] == bottom.shape[2], "top and bottom must have equal shape"

    a = top[3] / 255
    b = 1 - a

    return np.array([top[0]*a + bottom[0]*b, top[1]*a + bottom[1]*b, top[2]*a + bottom[2]*b])


def cleanUp(self):
    '''
    keep the path:list at a set length by:
        - delete coords in path that's from too long ago
        - keep a list of fully rendered layers from the delete point to the most recent render
        - update the object variable baseBG to the state at the delete point
    '''

    # trim list
    if len(self.raw_point_list) > self.path_limit:
        self.raw_point_list = self.raw_point_list[-(self.path_limit):]

    if len(self.rendered_layer_buffer) > self.path_limit:
        return self.rendered_layer_buffer.pop(0)
    else:
        return self.rendered_layer_buffer[0]


def loadBrushProfile(path):
    '''
    takes a path to an image
    returns a square 2D array 0-255
    '''
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = math.degrees(np.arctan2(y, x)) + 90
    return(rho, phi)


def defaltInp(dic, key, default_value):
    if key in dic.keys():
        if type(dic[key]) != type(default_value):
            print("WARNING: When initializing renderer process, render data of type {} does not match expected type of {}".format(type(dic[key]), type(default_value)))
        return dic[key]
    else:
        return default_value


def getStroke(raw_point_list):
    # IN: list of raw points. OUT: path mask
    def points2path(points_list, Gaussian_coeff=10):
        
        def bresenham_get_line(start, end):

            """Bresenham's Line Algorithm
            Produces a list of tuples from start and end
        
            points1 = get_line((0, 0), (3, 4))
            [(3, 4), (2, 3), (1, 2), (1, 1), (0, 0)]
            """

            # Setup initial conditions
            x1, y1 = start
            x2, y2 = end
            dx = x2 - x1
            dy = y2 - y1
        
            # Determine how steep the line is
            is_steep = abs(dy) > abs(dx)
        
            # Rotate line
            if is_steep:
                x1, y1 = y1, x1
                x2, y2 = y2, x2
        
            # Swap start and end points if necessary and store swap state
            swapped = False
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                swapped = True
        
            # Recalculate differentials
            dx = x2 - x1
            dy = y2 - y1
        
            # Calculate error
            error = int(dx / 2.0)
            ystep = 1 if y1 < y2 else -1
        
            # Iterate over bounding box generating points between start and end
            y = y1
            points = []
            for x in range(x1, x2 + 1):
                coord = (y, x) if is_steep else (x, y)
                points.append(coord)
                error -= abs(dy)
                if error < 0:
                    y += ystep
                    error += dx
        
            # Reverse the list if the coordinates were swapped
            if swapped:
                points.reverse()
            return points


        path = []
        for i, point in enumerate(raw_point_list):
            if i == len(raw_point_list) - 1 or len(raw_point_list) == 0:
                break # to avoid out of bound or first dot
            else:
                path += bresenham_get_line(point, raw_point_list[i+1])

        #TODO add Gaussian smooth to the list of points
        #TODO temporal decay
        # print(path) # should be a flat 2D list of pixels coord

        return np.array(path)

    #just for test
    return points2path(raw_point_list)