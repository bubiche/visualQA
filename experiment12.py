import os
import sys

for filename in os.listdir('videos'):
    if filename.endswith('.mp4'):
        vid_path = os.path.join('videos', filename)
        out_path = os.path.join('out_vid', filename.replace('.mp4', '.avi'))
        
        if filename[0] == 'p': cls_name = 'person'
        elif filename[0] == 'b': cls_name = 'bird'
        my_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=620 --cls={} --config=2 --see_vid="{}" --output_file_vid="{}"'.format(cls_name, vid_path, out_path)
        os.system(my_cmd)