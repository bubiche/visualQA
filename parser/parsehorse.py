from image2vec import yolo
import h5py
import os
import os.path

def file_len(fname):
    try:
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    except OSError:
        return 0
    


print('create hdf5 file')
img_vec_file = h5py.File('horses_vec.hdf5', 'w')
count_file = h5py.File('horses_count.hdf5', 'w')

img_list = [f for f in os.listdir('.') if f.endswith('.jpg')]

count_list = []
print('Create count')
for img in img_list:
    file_name_pre = os.path.splitext(img)[0]
    name_list = [file_name_pre, '_entires.groundtruth']
    file_name = ''.join(name_list)
    if os.path.isfile(file_name):
        count_list.append(0)
    else:
        count_list.append(file_len(file_name))

count_dset = count_file.create_dataset('count', data=count_list)
print('Count shape')
print(count_dset.shape)

print('Load YOLO...')
net = yolo.YOLO(
    'image2vec/yolo-small.cfg', 
    'image2vec/yolo-small.weights',
    up_to = 29)

img_dset = img_vec_file.create_dataset('vec', (340, 512), dtype='f')
i = 0
for img in img_list:
    print(i)
    vec = net.forward([img])
    i += 1
    img_dset[i] = vec[0]

 
img_vec_file.close()
count_file.close()