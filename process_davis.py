import numpy as np
import cv2
import os
import math
import skvideo.io

seq_path = '../dogs-jump'
img_list = [os.path.join(seq_path, x) for x in os.listdir(seq_path) if x.endswith('jpg')]

img_list.sort()
writer = skvideo.io.FFmpegWriter('out.avi', outputdict={'-b': '300000000'})

for img in img_list:
    frame = cv2.imread(img)
    frame = frame[:,:,::-1]
    frame = cv2.resize(frame, (448, 448))
    writer.writeFrame(frame)