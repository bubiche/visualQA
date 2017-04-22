import h5py
import sys

img_id = int(sys.argv[1])
f = h5py.File('parser/full_split_voc.hdf5', 'r')
d = f['test']
idx = d[img_id]
f.close()
f = h5py.File('parser/full_name_voc.hdf5', 'r')
d = f ['name']
fn = d[idx]
print(fn.decode())