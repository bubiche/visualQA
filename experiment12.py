import os
import sys

for filename in os.listdir('videos'):
    if filename.endswith('.mp4'):
        vid_path = os.path.join('videos', filename)
        out_path = os.path.join('out_vid', filename.replace('.mp4', '.avi'))
        
        my_cmd = '/home/tmbao_1995/miniconda3/bin/python main.py --load=620 --cls=person --config=2 --see_vid="{}" --output_file_vid="{}"'.format(vid_path, out_path)
        os.system(my_cmd)