import h5py

def merge_dat(f1, f2, dset_name, out_file):
    input_file_1 = h5py.File(f1, 'r')
    input_file_2 = h5py.File(f2, 'r')
    output_file = h5py.File(out_file, 'w')
    
    dset1 = input_file_1[dset_name]
    dset2 = input_file_2[dset_name]
    
    total_size = dset1.shape[0] + dset2.shape[0]
    
    if dset_name == 'vec':
        out_dset = output_file.create_dataset(dset_name, (total_size, 14, 14, 512), dtype='f')
    elif dset_name == 'count':
        out_dset = output_file.create_dataset(dset_name, (total_size,), dtype='i')
    else:
        return
    
    i = 0
    for dat in dset1:
        out_dset[i] = dat
        i += 1
        
    for dat in dset2:
        out_dset[i] = dat
        i += 1
        
    output_file.close()
    input_file_1.close()
    input_file_2.close()

merge_dat('horses_vec.hdf5', 'voc_vec.hdf5', 'vec', 'full_vec.hdf5')
merge_dat('horses_count.hdf5', 'voc_count.hdf5', 'count', 'full_count.hdf5')