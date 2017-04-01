import numpy as np

class BatchYielder(object):

    def __init__(self, batch_num, epoch, text_path, video_path):
        self.batch_num = batch_num
        self.epoch = epoch
        self.text_path = text_path
        self.video_path = video_path
        self.data_size = self.get_data_size()
        if self.batch_num > self.data_size: self.batch_num = self.data_size
        self.batch_size = int(self.data_size/self.batch_num)

    def get_data_size(self):
        # TODO: get data size

    def shuffle_data(self):
        self.shuffle_idx = np.random.permutation(self.data_size)

    def get_vector_for_text_at_index(self, idx):
        # TODO: get feature vector of text at index idx

    def get_vector_for_video_at_index(self, idx):
        # TODO: get feature vector of video at index idx

    def get_annotation_at_index(self, idx):
        #TODO: get the annotation at index idx

    def next_batch(self):
        for i in range(self.epoch):
            print('epoch number %d' % i)
            self.shuffle_data()
            for b in range(self.batch_num):
                # yield these
                x_batch = [[] for tmp in range(2)]
                y_batch = list()

                for j in range(b*self.batch_size, b*self.batch_size + batch_size):
                    if j >= self.data_size: continue
                    x_instance_text = self.get_vector_for_text_at_index(self.shuffle_idx[j])
                    if x_instance_text is None: continue
                    x_instance_video = self.get_vector_for_video_at_index(self.shuffle_idx[j])
                    y_instance = self.get_annotation_at_index(self.shuffle_idx[j])

                    x_batch[0].append(x_instance_text)
                    x_batch[1].append(x_instance_video)

                    y_batch.append(y_instance)

                yield x_batch, y_batch

