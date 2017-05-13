import numpy as np
import cv2
import os
import math
import skvideo.io

vid_path = 'catbirdvid'

vid_list = [os.path.join(vid_path, x) for x in os.listdir(vid_path) if x.endswith('mp4')]
vid_name_list = [x for x in os.listdir(vid_path) if x.endswith('mp4')]

vid_list.sort()
vid_name_list.sort()

print(vid_list)
print(vid_name_list)

i = 0
for vid in vid_list:
    videogen = skvideo.io.vreader(vid)
    metadata = skvideo.io.ffprobe(vid)
    frame_count = metadata['video']['@nb_frames']
    writer = skvideo.io.FFmpegWriter(vid_name_list[i], outputdict={
                                     '-b': '300000000', '-r': frame_rate})
                                     
    for frame in videogen:
        frame = cv2.resize(frame, (448, 448))
        writer.writeFrame(frame)
        
    writer.close()
    i += 1