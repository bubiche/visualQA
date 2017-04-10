import numpy as np
import h5py

class BatchYielder(object):

    def __init__(self, flags):
        self.batch_size = flags.batch_size
        self.epoch = flags.epoch
        self.vec_file = h5py.File(flags.vec_path, 'r')
        self.count_file = h5py.File(flags.count_path, 'r')
        self.vec_dset = self.vec_file['vec']
        self.count_dset = self.count_file['count']
        self.data_size = self.get_data_size()
        self.n_use = flags.n_use
        if self.n_use < 1 or self.n_use > self.data_size: self.n_use = self.data_size
        self.n_val = int(self.n_use * flags.val_ratio / 100)
        self.n_test = int(self.n_use * flags.test_ratio / 100)
        self.train_size = self.n_use - self.n_val - self.n_test
        self.shuffle_data()
        self.create_val_test_files()
        if self.batch_size > self.train_size: self.batch_size = self.train_size
        self.batch_per_epoch = int(np.ceil(self.train_size/self.batch_size))
        
    def create_val_test_files(self):
        self.val_vec = h5py.File('val_vec.hdf5', 'w')
        self.val_count = h5py.File('val_count.hdf5', 'w')
        self.test_vec = h5py.File('test_vec.hdf5', 'w')
        self.test_count = h5py.File('test_count.hdf5', 'w')
        
        self.val_vec_dset = self.val_vec.create_dataset('vec', (self.n_val, 7, 7, 1024), dtype='f')
        self.val_count_dset = self.val_vec.create_dataset('count', (self.n_val,), dtype='i')
        self.test_vec_dset = self.test_vec.create_dataset('vec', (self.n_test, 7, 7, 1024), dtype='f')
        self.test_count_dset = self.test_vec.create_dataset('count', (self.n_test,), dtype='i')
        
        i = 0
        while i < self.n_val:
            self.val_vec_dset[i] = self.get_x_at_index(self.shuffle_idx[i + self.train_size])
            self.val_count_dset[i] = self.get_annotation_at_index(self.shuffle_idx[i + self.train_size])
            i += 1
            print('val: %d' % (i))
            
        i = 0
        while i < self.n_test:
            self.test_vec_dset[i] = self.get_x_at_index(self.shuffle_idx[i + self.train_size + self.n_val])
            self.test_count_dset[i] = self.get_annotation_at_index(self.shuffle_idx[i + self.train_size + self.n_val])
            i += 1
            print('test: %d' % (i))
            
        self.val_vec.close()
        self.val_count.close()
        self.test_vec.close()
        self.test_count.close()

        self.val_vec = h5py.File('val_vec.hdf5', 'r')
        self.val_count = h5py.File('val_count.hdf5', 'r')
        self.test_vec = h5py.File('test_vec.hdf5', 'r')
        self.test_count = h5py.File('test_count.hdf5', 'r')        
        
        self.val_vec_dset = self.val_vec['vec']
        self.val_count_dset = self.val_vec['count']
        self.test_vec_dset = self.test_vec['vec']
        self.test_count_dset = self.test_vec['count']
        
    def get_data_size(self):
        return self.vec_dset.shape[0]

    def shuffle_data(self):
        self.shuffle_idx = np.random.permutation(self.data_size)
        
    def shuffle_train(self):
        self.shuffle_idx_train = np.random.permutation(self.train_size)

    def get_x_at_index(self, idx):
        return self.vec_dset[idx]

    def get_annotation_at_index(self, idx):
        return self.count_dset[idx]

    def get_x_train_at_index(self, idx):
        return self.get_x_at_index(self.shuffle_idx[idx])
        
    def get_annotation_train_at_index(self, idx):
        return self.get_annotation_at_index(self.shuffle_idx[idx])
        
    def next_batch(self):
        for i in range(self.epoch):
            print('epoch number %d' % i)
            self.shuffle_train()
            for b in range(self.batch_per_epoch):
                # yield these
                x_batch = list()
                y_batch = list()

                for j in range(b*self.batch_size, b*self.batch_size + self.batch_size):
                    if j >= self.train_size: continue
                    x_instance = self.get_x_train_at_index(self.shuffle_idx_train[j])
                    if x_instance is None: continue
                    y_instance = self.get_annotation_train_at_index(self.shuffle_idx_train[j])

                    x_batch.append(x_instance)
                    y_batch.append(y_instance)

                yield x_batch, y_batch
                
    def validation_set(self):
        ret_x = np.zeros((self.n_val, 7, 7, 1024))
        ret_y = np.zeros((self.n_val, 1024))
        
        for i in range(n_val):
            ret_x[i] = self.val_vec_dset[i]
            ret_y[i] = self.val_count_dset[i]
            
        return ret_x, ret_y
        
    def test_set(self):
        ret_x = np.zeros((self.n_test, 7, 7, 1024))
        ret_y = np.zeros((self.n_test, 1024))
        
        for i in range(n_test):
            ret_x[i] = self.test_vec_dset[i]
            ret_y[i] = self.test_count_dset[i]
            
        return ret_x, ret_y
