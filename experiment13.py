import os
import sys

for filename in os.listdir('catbirdvid'):
    if filename.endswith('.mp4'):
        vid_path = os.path.join('catbirdvid', filename)
        out_path = os.path.join('out_vid', filename.replace('.mp4', '.avi'))
        
        cls_name = 'cat'
        my_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=620 --cls={} --config=2 --see_vid="{}" --output_file_vid="{}"'.format(cls_name, vid_path, out_path)
        os.system(my_cmd)