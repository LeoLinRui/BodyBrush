from modules import renderer, Path
#from utils import alphaBlend

import time
import numpy as np
from multiprocessing import Queue, Process

import cv2

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
    
    from model import getPoints
    
    point_list_queue = Queue()
    inference_process = Process(target=getPoints, args=(point_list_queue,))
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
        print("time cost of retrieving inference result", time.time() - t_start)
        t_start = time.time()

#===============================================================================================================================

        for pt in point_list:
            master_dict[int(pt[0])].addPoint([int(pt[1]), int(pt[2])])

        # loop through all 17 items to: get render parameters from Path and put into inp q
        for i, path in enumerate(master_dict.values()):
            render_data = path.renderData()
            in_q.put([i, render_data])

        print("time cost of qing data", (time.time() - t_start))
        t_start = time.time()

        # perform alpha blending to the layer according to the redering sequence
        
        bottom_layer = np.transpose(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), [2, 1, 0])

        if counter != 0:
            for i in rendering_sequence:
                bottom_layer = alphaBlend(rendered_layers[i], bottom_layer)
            
        bottom_layer = np.transpose(bottom_layer, [2, 1, 0])
        bottom_layer = bottom_layer.astype(np.uint8)

        print("time cost of blending", (time.time() - t_start))



        rendered_img = cv2.cvtColor(bottom_layer, cv2.COLOR_HSV2RGB)

        for dot in point_list:
            img = cv2.circle(img, (int(dot[1]), int(dot[2])), radius=10, color=(0, 0, 255), thickness=-1)
        
        img = cv2.putText(img, str(fps)+"fps", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)

        cv2.imshow("image", img)
        cv2.imshow("drawimg", rendered_img)
        cv2.waitKey(1)
        
        counter += 1

        print("time from capture to display: ", time.time()-ovall_start)



        t_start = time.time()

        rendered_layers = list(range(17)) # will be in the numerical indexing order
        
        # retrieve rendered layers from out q and put into rendered_layers for blending
        for _ in range(len(master_dict)):
            rendered_layer = out_q.get()
            rendered_layers[rendered_layer[0]] = rendered_layer[1]

        print("1 frame is done: time cost (on top of parallel) of rendering", (time.time() - t_start), "len", len(rendering_sequence))
        print("overall cost of one frame: ", time.time()-ovall_start)

#===============================================================================================================================