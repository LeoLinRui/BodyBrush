import multiprocessing
from model import getPoints
from modules import PathManager

import time
import numpy as np
import multiprocessing

import cv2


if __name__ == '__main__':

    #start a queue and spwan the rcnn inference process
    point_list_queue = multiprocessing.Queue()
    inference_process = multiprocessing.Process(target=getPoints, args=(point_list_queue,))
    inference_process.start()

    path_manager = PathManager([1000, 1000])

    t_start = 0 # time for fps

    print("starting main loop")

    while True:

        fps = int(1 / (time.time() - t_start))
        t_start = time.time()

        img, point_list = point_list_queue.get()

        path_manager.addPoints(point_list)
        rendered_img = path_manager.render()

        print(rendered_img.shape)

        for dot in point_list:
            img = cv2.circle(img, (int(dot[1]), int(dot[2])), radius=10, color=(0, 0, 255), thickness=-1)
        
        img = cv2.putText(img, str(fps)+"fps", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (255, 0, 0), thickness=2)

        cv2.imshow("image", img)
        # cv2.imshow("drawing", rendered_img)
        cv2.waitKey(1)