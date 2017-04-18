import numpy as np
import h5py
import random

class BatchYielderBinhYen(object):

    def __init__(self, flags):
        self.batch_size = flags.batch_size
        self.epoch = flags.epoch
        
        self.vec_file = h5py.File(flags.voc_vec_path, 'r')
        self.count_file = h5py.File(flags.voc_count_path, 'r')
        self.split_file = h5py.File(flags.split_path, 'r')
        
        self.vec_dset = self.vec_file['vec']
        self.count_dset = self.count_file[flags.cls]
        self.train_dset = self.split_file['train']
        self.val_dset = self.split_file['test']
        self.test_dset = self.split_file['test']
        
        self.n_train = self.train_dset.shape[0]
        self.n_val = self.val_dset.shape[0]
        self.n_test = self.test_dset.shape[0]
        
        if self.batch_size > self.n_train: self.batch_size = self.n_train
        self.batch_per_epoch = int(np.ceil(self.n_train/self.batch_size))
        
    def shuffle_data(self):
        self.shuffle_idx = np.random.permutation(self.n_train)
            
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
                    if j >= self.n_train: continue
                    x_instance = self.get_x_at_index(self.train_dset[self.shuffle_idx[j]])
                    y_instance = self.get_annotation_at_index(self.train_dset[self.shuffle_idx[j]])

                    x_batch.append(x_instance)
                    y_batch.append(y_instance)

                yield x_batch, y_batch
                
    def next_epoch(self):
        x_batch = list()
        y_batch = list()

        for j in range(self.n_train):
            x_instance = self.get_x_at_index(self.train_dset[self.shuffle_idx[j]])
            y_instance = self.get_annotation_at_index(self.train_dset[self.shuffle_idx[j]])

            x_batch.append(x_instance)
            y_batch.append(y_instance)

        return np.array(x_batch), np.array(y_batch)
        
    def validation_set(self):
        ret_x = np.zeros((self.n_val, 7, 7, 1024))
        ret_y = np.zeros((self.n_val,))
        
        for i in range(self.n_val):
            ret_x[i] = self.get_x_at_index(self.val_dset[i])
            ret_y[i] = self.get_annotation_at_index(self.val_dset[i])
            
        return ret_x, ret_y
        
    def test_set(self):
        ret_x = np.zeros((self.n_test, 7, 7, 1024))
        ret_y = np.zeros((self.n_test,))
        
        for i in range(self.n_test):
            ret_x[i] = self.get_x_at_index(self.test_dset[i])
            ret_y[i] = self.get_annotation_at_index(self.test_dset[i])
            
        return ret_x, ret_y