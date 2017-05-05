import numpy as np
import cv2
import os
import math
from .image2vec import yolo
import skvideo.io

class VideoDemo(object):
    def __init__(self, flags, net):
        self.net = net
        self.feature_extractor = yolo.YOLO(
            'parser/image2vec/yolo-full.cfg', 
            'parser/image2vec/yolo-full.weights',
            up_to = 28)
        self.conf_id = flags.config
        self.cls = flags.cls
        self.vid_path = flags.see_vid
        self.output_file = flags.output_file_vid

    def process_frame(self, frame, att_vec, predict_count):
        resized_image = cv2.resize(frame, (448, 448))
        res = np.array(resized_image)
        res = res * att_vec[:,:, None]
        cv2.putText(res, '{} {}(s)'.format(predict_count, self.cls), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        return res
        
    def seek_vid(self):
        assert os.path.isfile(self.vid_path), 'file {} does not exist'.format(self.vid_path)
        videogen = skvideo.io.vreader(self.vid_path)
        writer = skvideo.io.FFmpegWriter(self.output_file)
        for frame in videogen:
            vec = self.feature_extractor.forward_frame(frame)
            att_vec, predict_count = self.net.get_interpolated_attention(vec, 448)
            frame = self.process_frame(frame, att_vec[0], predict_count)
            writer.writeFrame(frame)
            
        writer.close()