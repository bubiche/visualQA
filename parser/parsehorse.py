from image2vec import yolo
import h5py
import os
import os.path

def file_len(fname):
    with open(fname) as myfile:
        count = sum(1 for line in myfile)
        
    return count

print('create hdf5 file')
img_vec_file = h5py.File('horses_vec_23.hdf5', 'w')
count_file = h5py.File('horses_count_23.hdf5', 'w')

img_list = [f for f in os.listdir('.') if f.endswith('.jpg')]

count_list = []
print('Create count')
for img in img_list:
    file_name_pre = os.path.splitext(img)[0]
    name_list = [file_name_pre, '_entires.groundtruth']
    file_name = ''.join(name_list)
    if not os.path.isfile(file_name):
        count_list.append(0)
    else:
        print(file_len(file_name))
        count_list.append(file_len(file_name))

count_dset = count_file.create_dataset('count', data=count_list)
print('Count shape')
print(count_dset.shape)

print('Load YOLO...')
net = yolo.YOLO(
    'image2vec/yolo-full.cfg', 
    'image2vec/yolo-full.weights',
    up_to = 23)

img_dset = img_vec_file.create_dataset('vec', (340, 14, 14, 512), dtype='f')
i = 0
for img in img_list:
    vec = net.forward([img])
    img_dset[i] = vec[0]
    if i % 100 == 0:
        print(i)
    i += 1

 
img_vec_file.close()
count_file.close()