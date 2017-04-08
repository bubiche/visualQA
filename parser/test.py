import skvideo.io

BASE_VID_PATH = '/home/npnguyen/ugvideo/train_videos/'

def get_vid_path_from_vid_id(vid_id):
    path_list = [BASE_VID_PATH, vid_id, '.mp4']
    return ''.join(path_list)
    
filename = get_vid_path_from_vid_id('video1')
metadata = skvideo.io.ffprobe(filename)
print(metadata['video']['@nb_frames'])