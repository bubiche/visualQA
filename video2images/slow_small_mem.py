import skvideo.io
import os

EXTRACT_FRAME_COUNT = 150

def extract_images(cap, dir, step, skip, verbose = True):
    count = 0
    count_save = 0
    for frame in cap:
        if skip > 0:
            skip -= 1
            continue
            
        if count % step == 0:
            skvideo.io.vwrite("%s/frame%d.bmp" % (dir, count_save), frame)
            count_save += 1
            
            if verbose:
                print('Tinh yeu dam da %d/%d' % (count_save, EXTRACT_FRAME_COUNT))
            
        if count_save == EXTRACT_FRAME_COUNT:
            return
        
        count += 1

for filename in os.listdir('.'):
    if filename.endswith('.mp4'):
        cap = skvideo.io.vreader(filename)
        metadata = skvideo.io.ffprobe(filename)
        frame_count = int(metadata['video']['@nb_frames'])
        
        step = 0
        skip = 0
        if frame_count < EXTRACT_FRAME_COUNT:
            step = 1
        else:
            step = int(frame_count/EXTRACT_FRAME_COUNT)
            skip = int((frame_count - step * EXTRACT_FRAME_COUNT)/2) - 1
            
        dir = (os.path.splitext(filename)[0])
        if not os.path.exists(dir):
            os.makedirs(dir)

        print('Cuoc tinh nay da lo trao ve em %s' % filename)
        extract_images(cap, dir, step, skip, True)
        cap.close()
