import h5py
import numpy as np
import cv2
import os

class Visualizer(object):
    def __init__(self, flags):
        self.vec_file = h5py.File(flags.full_vec_path, 'r')
        self.name_file = h5py.File(flags.full_name_path, 'r')
        
        self.vec_dset = self.vec_file['vec']
        self.name_dset = self.name_file['name']
        
        self.file_path = flags.see_path
        self.img_list = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path) if f.endswith('.jpg')]
        
    def get_vec(self, my_img_path):
        print('Searching for images')
        i = 0
        head, tail = os.path.split(my_img_path)
        file_name = tail
        for i in range(int(self.name_dset.shape[0])):
            if self.name_dset[i].decode() == file_name:
                return self.vec_dset[i]
                
        print('Cannot find %s' % (file_name))
        return np.zeros((7, 7, 1024))
        
    def get_vecs(self):
        ret = np.zeros((len(self.img_list), 7, 7, 1024))
        
        i = 0
        print(self.img_list)
        for img in self.img_list:
            print(img)
            ret[i] = self.get_vec(img)
            i += 1
            
        return ret
        
    def visualize(self, att_vec, file_path, idx):
        '''
        att_vec is (7, 7)
        '''
        img = cv2.imread(file_path)
        resized_image = cv2.resize(img, (448, 448))
        numrows, numcols = 7, 7
        height = int(resized_image.shape[0] / numrows)
        width = int(resized_image.shape[1] / numcols)
        
        res = np.array(resized_image)
        for row in range(numrows):
            for col in range(numcols):
                y0 = row * height
                y1 = y0 + height
                x0 = col * width
                x1 = x0 + width
                res[y0:y1, x0:x1] = res[y0:y1, x0:x1] * 2 * att_vec[row][col]
                
        output_file = '{}.jpg'.format(idx)
        cv2.imwrite(output_file, res.astype(np.uint8))
        
    def visualize_multiple(self, att_vec):
        i = 0
        for img in self.img_list:
            self.visualize(att_vec[i], img, i)