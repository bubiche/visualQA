import skvideo.io
import os

EXTRACT_FRAME_COUNT = 150

def extract_images(cap, n, directory, step, skip, verbose = True):
    count_save = 0
    i = skip

    while i < n:
        skvideo.io.vwrite("%s/frame%d.bmp" % (directory, count_save), cap[i])
        count_save += 1    
            
        if verbose:
            print('Tinh yeu dam da %d/%d' % (count_save, EXTRACT_FRAME_COUNT))
            
        if count_save == EXTRACT_FRAME_COUNT:
            return
        
        i += step

for filename in os.listdir('.'):
    if filename.endswith('.mp4'):
        cap = skvideo.io.vread(filename)
        metadata = skvideo.io.ffprobe(filename)
        frame_count = int(metadata['video']['@nb_frames'])
        
        step = 0
        skip = 0
        if frame_count < EXTRACT_FRAME_COUNT:
            step = 1
        else:
            step = int(frame_count/EXTRACT_FRAME_COUNT)
            skip = int((frame_count - step * EXTRACT_FRAME_COUNT)/2) - 1
            
        directory = (os.path.splitext(filename)[0])
        if not os.path.exists(directory):
            os.makedirs(directory)

        print('Cuoc tinh nay da lo trao ve em %s' % filename)
        extract_images(cap, frame_count, directory, step, skip, True)
