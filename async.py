from modules import renderer, Path
#from utils import alphaBlend

import time
import numpy as np
from multiprocessing import Queue, Process

import cv2
import os

'''def alphaBlend(top, bottom):
    
    # takes an HSLA top and a bottom, blends and returns a HSL image
    
    #assert top.shape[0] == 4, "top must have alpha channel"
    #assert top.shape[1] == bottom.shape[1], "top and bottom must have equal shape"
    #assert top.shape[2] == bottom.shape[2], "top and bottom must have equal shape"

    foreground = top[0:3].astype(float)
    background = bottom.astype(float)

    s = time.time()
    alpha = np.stack((top[3].astype(float), top[3].astype(float), top[3].astype(float))) /255
    print("time cost of stacking", time.time() - s)

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)

    return cv2.add(foreground, background)'''


def alphaBlend(top, bottom):
    '''
    takes an HSLA top and a bottom, blends and returns a HSL image
    '''
    #assert top.shape[0] == 4, "top must have alpha channel"
    #assert top.shape[1] == bottom.shape[1], "top and bottom must have equal shape"
    #assert top.shape[2] == bottom.shape[2], "top and bottom must have equal shape"

    s = time.time()
    
    a = np.divide(top[3].astype(float), 255)
    b = np.subtract(1, a)

    final = np.add(np.multiply(top[0:3], a), np.multiply(bottom, b)).astype(np.uint8)

    print("time cost of stacking", time.time() - s)

    return final


if __name__ == '__main__':

    res = [1000, 1000]
    master_dict = {} # {pathID:[path object, render process, input q, output q]} 
    BG = np.vstack([[np.ones(res)*179], [np.ones(res)*10], [np.ones(res)*100]])
    rendering_sequence = range(17) #change layer order here

    in_q = Queue()
    out_q = Queue()

    for i in range(8):
        process = Process(target=renderer, args=(in_q, out_q))
        process.start()

    for i in range(17):
        master_dict[i] = Path(map_resolution=res)

    print("PathManager initialization complete.")
    
#===============================================================================================================================

#start a queue and spwan the rcnn inference process
    
    from model import getPointsAsync
    
    point_list_queue = Queue()
    inference_process = Process(target=getPointsAsync, args=(point_list_queue,))
    inference_process.start()

#===============================================================================================================================

    t_start = 0 # time for fps
    print("starting main loop")

    counter = 0
    while True:
        ovall_start = time.time()
        fps = int(1 / (time.time() - t_start))
        t_start = time.time()

        img, point_list = point_list_queue.get()
#===============================================================================================================================

        for pt in point_list:
            master_dict[int(pt[0])].addPoint([int(pt[1]), int(pt[2])])

        # loop through all 17 items to: get render parameters from Path and put into inp q
        for i, path in enumerate(master_dict.values()):
            render_data = path.renderData()
            in_q.put([i, render_data])

        # perform alpha blending to the layer according to the redering sequence

        rendered_layers = list(range(17)) # will be in the numerical indexing order
        
        # retrieve rendered layers from out q and put into rendered_layers for blending
        for _ in range(len(master_dict)):
            rendered_layer = out_q.get()
            rendered_layers[rendered_layer[0]] = rendered_layer[1]


        bottom_layer = np.transpose(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), [2, 1, 0])

        for i in rendering_sequence:
            bottom_layer = alphaBlend(rendered_layers[i], bottom_layer)
            
        bottom_layer = np.transpose(bottom_layer, [2, 1, 0])
        bottom_layer = bottom_layer.astype(np.uint8)

        rendered_img = cv2.cvtColor(bottom_layer, cv2.COLOR_HSV2RGB)

        print("image blending complete")
        cv2.imwrite(os.path.join(r"I:\TRS Project 2\async_video_out", str(counter)+".jpg"), rendered_img)
        
        counter += 1

#===============================================================================================================================