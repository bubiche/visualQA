import h5py
import numpy as np
import cv2
import os
import math

class Visualizer(object):
    def __init__(self, flags, net):
        self.net = net
        self.vec_file = h5py.File(flags.full_vec_path, 'r')
        self.name_file = h5py.File(flags.full_name_path, 'r')
        self.full_voc_vec_file = h5py.File(flags.voc_vec_path, 'r')
        self.test_idx_file = h5py.File(flags.split_path, 'r')
        self.full_voc_name_file = h5py.File(flags.voc_name_path, 'r')
        
        self.vec_dset = self.vec_file['vec']
        self.name_dset = self.name_file['name']
        self.full_voc_vec = self.full_voc_vec_file['vec']
        self.test_idx = self.test_idx_file['test']
        self.full_voc_name = self.full_voc_name_file['name']
        
        self.file_path = flags.see_path
        self.img_list = [os.path.join(self.file_path, f) for f in os.listdir(self.file_path) if f.endswith('.jpg')]
        
    def get_vec(self, my_img_path):
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
            ret[i] = self.get_vec(img)
            i += 1
            
        return ret
        
    def visualize_xinhdep(self, att_vec, file_path, idx):
        '''
        att_vec is (7, 7)
        '''
        img = cv2.imread(file_path)
        resized_image = cv2.resize(img, (448, 448))
        numrows, numcols = 7, 7
        height = int(resized_image.shape[0] / numrows)
        width = int(resized_image.shape[1] / numcols)
        
        res = np.array(resized_image)
        centers = list()
        for row in range(numrows):
            for col in range(numcols):
                y0 = row * height
                y1 = y0 + height
                x0 = col * width
                x1 = x0 + width
                if att_vec[row][col] > 0.2:
                    centers.append((y0 + int((y1-y0)/2), x0 + int((x1-x0)/2)))
                    print((y0 + int((y1-y0)/2), x0 + int((x1-x0)/2)))
                #res[y0:y1, x0:x1] = res[y0:y1, x0:x1] * att_vec[row][col]
        
        # goes back to old method
        if len(centers) == 0:
            for row in range(numrows):
                for col in range(numcols):
                    y0 = row * height
                    y1 = y0 + height
                    x0 = col * width
                    x1 = x0 + width
                    res[y0:y1, x0:x1] = res[y0:y1, x0:x1] * att_vec[row][col]
                
        for c in centers:
            res = self.make_mask(res, 50, c)
            
        output_file = '{}.jpg'.format(idx)
        cv2.imwrite(output_file, res.astype(np.uint8))

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
        centers = list()
        for row in range(numrows):
            for col in range(numcols):
                y0 = row * height
                y1 = y0 + height
                x0 = col * width
                x1 = x0 + width
                weight = att_vec[row][col]
                if weight < 0.3333: weight = 0.3333
                res[y0:y1, x0:x1] = res[y0:y1, x0:x1] * weight
            
        output_file = '{}.jpg'.format(idx)
        cv2.imwrite(output_file, res.astype(np.uint8))
        
    def visualize_multiple(self, att_vec):
        i = 0
        for img in self.img_list:
            print(img)
            self.visualize(att_vec[i], img, i)
            i += 1
            
    def make_mask(self, image, radius, center=(0, 0)):
        r, c, d = image.shape
        res = np.array(image)
        for row in range(r):
            for col in range(c):
                if r == center[0] and c == center[1]:
                    weight = 1
                else:
                    dist_squared = (center[0] - row)**2 + (center[1] - col)**2
                    #dist = math.sqrt(dist_squared)
                    weight = math.exp((-0.5 * dist_squared)/(radius*radius))
                    if weight < 0.2:
                        weight = 0.2
                res[row][col] = res[row][col] * weight
        return res
        
    def visualize_test_set(self):
        self.all_img = list()
        self.all_vecs = np.zeros((self.test_idx.shape[0], 7, 7, 1024))

        i = 0
        for idx in self.test_idx:
            self.all_img.append(os.path.join('parser', self.full_voc_name[idx].decode()))
            self.all_vecs[i] = self.full_voc_vec[idx]
            i += 1
            
        att_vec = self.net.get_attention(self.all_vecs)
        
        i = 0
        for img in self.all_img:
            print(img)
            self.visualize(att_vec[i], img, i)
            i += 1