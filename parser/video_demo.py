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
        cv2.putText(res, '{} {}(s)'.format(predict_count, self.cls), (10,50), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,255,155), 5, cv2.LINE_AA)
        return res.astype(np.uint8)
        
    def seek_vid(self):
        assert os.path.isfile(self.vid_path), 'file {} does not exist'.format(self.vid_path)
        videogen = skvideo.io.vreader(self.vid_path)
        metadata = skvideo.io.ffprobe(self.vid_path)
        frame_rate = metadata['video']['@avg_frame_rate'].split('/')[0]
        frame_count = metadata['video']['@nb_frames']
        writer = skvideo.io.FFmpegWriter(self.output_file, outputdict={
                                         '-b': '300000000', '-r': frame_rate
                                        })
        i = 0
        count_out = 0
        for frame in videogen:
            vec = self.feature_extractor.forward_frame(frame)
            att_vec, predict_count = self.net.get_interpolated_attention(vec, 448)
            inp_frame = cv2.resize(frame, (448, 448))
            if i % 5 == 0:
                count_out = predict_count
            frame = self.process_frame(frame, att_vec[0], count_out)
            out_frame = np.concatenate((inp_frame, frame), 1)
            writer.writeFrame(out_frame)
            print('{}/{} frames'.format(i+1, frame_count))
            i += 1
            
        writer.close()
        
    def seek_vid_test(self):
        assert os.path.isfile(self.vid_path), 'file {} does not exist'.format(self.vid_path)
        videogen = skvideo.io.vreader(self.vid_path)
        metadata = skvideo.io.ffprobe(self.vid_path)
        frame_count = metadata['video']['@nb_frames']
        for frame in videogen:
            vec = self.feature_extractor.forward_frame(frame)
            att_vec, predict_count = self.net.get_attention(vec)
            
        print(frame_count)
