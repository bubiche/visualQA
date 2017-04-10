import numpy as np
import h5py
import random

class BatchYielder(object):

    def __init__(self, flags):
        self.batch_size = flags.batch_size
        self.epoch = flags.epoch
        
        self.vec_file = h5py.File(flags.vec_path, 'r')
        self.count_file = h5py.File(flags.count_path, 'r')
        self.vec_dset = self.vec_file['vec']
        self.count_dset = self.count_file['count']
        
        self.vec_val = h5py.File(flags.val_vec_path, 'r')
        self.count_val = h5py.File(flags.val_count_path, 'r')
        self.val_vec_dset = self.vec_val['vec']
        self.val_count_dset = self.count_val['count']
        
        self.vec_test = h5py.File(flags.test_vec_path, 'r')
        self.count_test = h5py.File(flags.test_count_path, 'r')
        self.test_vec_dset = self.vec_test['vec']
        self.test_count_dset = self.count_test['count']
        
        self.yes_train_dset = self.count_file['horse']
        print('Train set has %d images with horses' % (self.yes_train_dset[0]))
               
        self.n_val = self.val_count_dset.shape[0]
        self.n_test = self.test_count_dset.shape[0]
        self.train_size = self.yes_train_dset[0] * 2
        if self.batch_size > self.train_size: self.batch_size = self.train_size
        self.batch_per_epoch = int(np.ceil(self.train_size/self.batch_size))
        

    def get_data_size(self):
        return self.vec_dset.shape[0]

    def shuffle_data(self):
        self.shuffle_idx = list(range(self.yes_train_dset[0]))
        no_horse = random.sample(range(self.yes_train_dset[0], self.train_size), self.yes_train_dset[0])
        self.shuffle_idx.extend(no_horse)
        np.random.shuffle(self.shuffle_idx)

    def get_x_at_index(self, idx):
        return self.vec_dset[idx]

    def get_annotation_at_index(self, idx):
        return self.count_dset[idx]
        
    def next_batch(self):
        for i in range(self.epoch):
            print('epoch number %d' % i)
            self.shuffle_data()
            for b in range(self.batch_per_epoch):
                # yield these
                x_batch = list()
                y_batch = list()

                for j in range(b*self.batch_size, b*self.batch_size + self.batch_size):
                    if j >= self.train_size: continue
                    x_instance = self.get_x_at_index(self.shuffle_idx[j])
                    if x_instance is None: continue
                    y_instance = self.get_annotation_at_index(self.shuffle_idx[j])

                    x_batch.append(x_instance)
                    y_batch.append(y_instance)

                yield x_batch, y_batch
                
    def validation_set(self):
        ret_x = np.zeros((self.n_val, 7, 7, 1024))
        ret_y = np.zeros((self.n_val,))
        
        for i in range(self.n_val):
            ret_x[i] = self.val_vec_dset[i]
            ret_y[i] = self.val_count_dset[i]
            
        return ret_x, ret_y
        
    def test_set(self):
        ret_x = np.zeros((self.n_test, 7, 7, 1024))
        ret_y = np.zeros((self.n_test,))
        
        for i in range(self.n_test):
            ret_x[i] = self.test_vec_dset[i]
            ret_y[i] = self.test_count_dset[i]
            
        return ret_x, ret_y
