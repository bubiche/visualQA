import h5py

f = h5py.File('full_split_voc.hdf5', 'r')
d = f['test']
idx = d[2451]
f.close()
f = h5py.File('full_name_voc.hdf5', 'r')
d = f ['name']
fn = d[idx]
print(fn.decode())