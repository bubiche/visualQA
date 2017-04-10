import numpy as np
import h5py

COUNT = 2538
class Splitter(object):

    def __init__(self):
        self.vec_file = h5py.File('full_vec.hdf5', 'r')
        self.count_file = h5py.File('full_count.hdf5', 'r')
        self.vec_dset = self.vec_file['vec']
        self.count_dset = self.count_file['count']
        self.data_size = self.get_data_size()
        self.n_val = int(self.n_use * 15 / 100)
        self.n_test = int(self.n_use * 15 / 100)
        self.n_train = self.data_size - self.n_val - self.n_test
        self.shuffle_data()
        
    def save_train_val_test()
        self.val_vec = h5py.File('val_vec.hdf5', 'w')
        self.val_count = h5py.File('val_count.hdf5', 'w')
        self.test_vec = h5py.File('test_vec.hdf5', 'w')
        self.test_count = h5py.File('test_count.hdf5', 'w')
        self.train_vec = h5py.File('train_vec_tmp.hdf5', 'w')
        self.train_count = h5py.File('train_count_tmp.hdf5', 'w')
        
        self.val_vec_dset = self.val_vec.create_dataset('vec', (self.n_val, 7, 7, 1024), dtype='f')
        self.val_count_dset = self.val_count.create_dataset('count', (self.n_val,), dtype='i')
        self.test_vec_dset = self.test_vec.create_dataset('vec', (self.n_test, 7, 7, 1024), dtype='f')
        self.test_count_dset = self.test_count.create_dataset('count', (self.n_test,), dtype='i')
        self.train_vec_dset_tmp = self.train_vec.create_dataset('vec', (self.n_train, 7, 7, 1024), dtype='f')
        self.train_count_dset_tmp = self.train_count.create_dataset('count', (self.n_train,), dtype='i')
        
        print('Dump train')
        i = 0
        while i < self.n_train:
            self.train_vec_dset_tmp[i] = self.get_x_at_index(self.shuffle_idx[i])
            self.train_count_dset_tmp[i] = self.get_annotation_at_index(self.shuffle_idx[i])
            i += 1
        
        print(self.train_vec_dset_tmp.shape)
        print(self.train_count_dset_tmp.shape)
        
        print('Dump val')
        i = 0
        yes = 0
        while i < self.n_val:
            self.val_vec_dset[i] = self.get_x_at_index(self.shuffle_idx[i + self.n_train])
            self.val_count_dset[i] = self.get_annotation_at_index(self.shuffle_idx[i + self.n_train])
            if val_count_dset[i] == 0: yes += 1
            i += 1
        
        print(self.val_vec_dset.shape)
        print(self.val_count_dset.shape)
        print('yes : %d' % (yes))
        
        print('Dump test')
        i = 0
        yes = 0
        while i < self.n_test:
            self.test_vec_dset[i] = self.get_x_at_index(self.shuffle_idx[i + self.n_train + self.n_val])
            self.test_count_dset[i] = self.get_annotation_at_index(self.shuffle_idx[i + self.n_train + self.n_val])
            if test_count_dset[i] == 0: yes += 1
            i += 1
         
        print(self.test_vec_dset.shape)
        print(self.test_count_dset.shape)
        print('yes : %d' % (yes))
        
        self.val_vec.close()
        self.val_count.close()
        self.test_vec.close()
        self.test_count.close()
        self.train_vec.close()
        self.train_count.close()
        
    def get_data_size(self):
        return self.vec_dset.shape[0]

    def shuffle_data(self):
        self.shuffle_idx = np.random.permutation(self.data_size)
        
    def shuffle_train(self):
        self.shuffle_idx_train = np.random.permutation(self.n_train)

    def get_x_at_index(self, idx):
        return self.vec_dset[idx]

    def get_annotation_at_index(self, idx):
        return self.count_dset[idx]
    
    def restructure_train(self):
        train_vec_tmp = h5py.File('train_vec_tmp.hdf5', 'r')
        train_count_tmp = h5py.File('train_count_tmp.hdf5', 'r')
        train_vec = h5py.File('train_vec.hdf5', 'w')
        train_count = h5py.File('train_count.hdf5', 'w')
        
        train_vec_dset_tmp = train_vec_tmp['vec']
        train_count_dset_tmp = train_count_tmp['count']
        train_vec_dset = train_vec.create_dataset('vec', (self.n_train, 7, 7, 1024), dtype='f')
        train_count_dset = train_count.create_dataset('count', (self.n_train,), dtype='i')
        yes_count_dset = train_count.create_dataset('horse', (1,), dtype='i')
        
        count_arr = np.zeros((self.n_train,))
        count_arr[:] = train_count_dset_tmp[:]
        new_idx = [i[0] for i in sorted(list(enumerate(count_arr)), key=lambda x:x[1], reverse=True)]
        count_arr[::-1].sort()
        
        print('Max: %d' % (int(count_arr[0])))
        yes_count = 0
        for i in range(self.n_train):
            train_count_dset[i] = int(count_arr[i])
            train_vec_dset[i] = train_vec_dset_tmp[new_idx[i]]
            if train_count_dset[i] > 0:
                yes_count += 1
                
        yes_count_dset[0] = yes_count
        print('Yes count: %d' % (yes_count))
            
        train_vec_tmp.close()
        train_count_tmp.close()
        train_vec.close()
        train_count.close()
        