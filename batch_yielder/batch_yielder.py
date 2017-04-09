import numpy as np
import h5py

class BatchYielder(object):

    def __init__(self, batch_size, epoch, vec_path, count_path, n_use):
        '''
        n_use = 1439 will have all horse images + 170 no horse
        n_use =2538 will have yes == no
        '''
        self.batch_size = batch_size
        self.epoch = epoch
        self.vec_file = h5py.File(vec_path, 'r')
        self.count_file = h5py.File(count_path, 'r')
        self.vec_dset = self.vec_file['vec']
        self.count_dset = self.count_file['count']
        self.data_size = self.get_data_size()
        self.n_use = n_use
        if self.n_use < 1 or self.n_use > self.data_size: self.n_use = self.data_size
        self.data_size = self.n_use
        if self.batch_size > self.data_size: self.batch_size = self.data_size
        self.batch_per_epoch = int(np.ceil(self.data_size/self.batch_size))

    def __del__(self):
        self.vec_file.close()
        self.count_file.close()

    def get_data_size(self):
        return self.vec_dset.shape[0]

    def shuffle_data(self):
        self.shuffle_idx = np.random.permutation(self.data_size)

    def get_x_at_index(self, idx):
        return vec_dset[idx]

    def get_annotation_at_index(self, idx):
        return count_dset[idx]

    def next_batch(self):
        for i in range(self.epoch):
            print('epoch number %d' % i)
            self.shuffle_data()
            for b in range(self.batch_per_epoch):
                # yield these
                x_batch = list()
                y_batch = list()

                for j in range(b*self.batch_size, b*self.batch_size + self.batch_size):
                    if j >= self.data_size: continue
                    x_instance = self.get_x_at_index(self.shuffle_idx[j])
                    if x_instance_text is None: continue
                    y_instance = self.get_annotation_at_index(self.shuffle_idx[j])

                    x_batch.append(x_instance)
                    y_batch.append(y_instance)

                yield x_batch, y_batch

