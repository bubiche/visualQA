import numpy as np        
import cv2
import h5py
import os
import random
from image2vec import yolo

class Image_Transformer(object):
    def __init__(self, flags):
        self.name_file = h5py.File(flags.full_name_path, 'r')
        self.full_vec_file = h5py.File(flags.full_vec_path, 'r')
        self.full_count_file = h5py.File(flags.full_count_path, 'r')
        self.horses_file = h5py.File('parser/all_horse.hdf5', 'r')
        
        self.net = net = yolo.YOLO(
            'parser/image2vec/yolo-full.cfg', 
            'parser/image2vec/yolo-full.weights',
            up_to = 28)
        if flags.equal:
            self.n_additional_random = 861
        else:
            self.n_additional_random = 141
            
        self.name_dset = self.name_file['name']
        self.vec_dset = self.full_vec_file['vec']
        self.horse_name_dset = self.horses_file['name']
        self.horse_count_dset = self.horses_file['count']
        self.count_dset = self.full_count_file['count']
        
        self.data_path = flags.data_path
        
    def get_transformed_vec(self, vec):
        for i in range(int(self.vec_dset.shape[0])):
            if np.allclose(self.vec_dset[i], vec):
                return self.transform_vec(i)
                
        return vec
        
    def imcv2_recolor(self, im, a = .1):
        t = [np.random.uniform()]
        t += [np.random.uniform()]
        t += [np.random.uniform()]
        t = np.array(t) * 2. - 1.

        # random amplify each channel
        im = im * (1 + t * a)
        mx = 255. * (1 + a)
        up = np.random.uniform() * 2 - 1
        im = np.power(im/mx, 1. + up * .5)
        return np.array(im * 255., np.uint8)

    def imcv2_affine_trans(self, im):
        # Scale and translate
        h, w, c = im.shape
        scale = np.random.uniform() / 20. + 1.
        max_offx = (scale-1.) * w
        max_offy = (scale-1.) * h
        offx = int(np.random.uniform() * max_offx)
        offy = int(np.random.uniform() * max_offy)
    
        im = cv2.resize(im, (0,0), fx = scale, fy = scale)
        im = im[offy : (offy + h), offx : (offx + w)]
        flip = np.random.binomial(1, .5)
        if flip: im = cv2.flip(im, 1)
        return im
    
    def transform_vec(self, idx):
        img_path = os.path.join(self.data_path, self.name_dset[idx].decode())
        img = cv2.imread(img_path)
        img = cv2.resize(img, (448, 448))
        img = self.imcv2_affine_trans(img)
        img = self.imcv2_recolor(img)
        
        img = np.clip(img, 0, 255)
        
        return img
        
    def transform_img(self, pth):
        img = cv2.imread(pth)
        img = cv2.resize(img, (448, 448))
        img = self.imcv2_affine_trans(img)
        img = self.imcv2_recolor(img)
        
        img = np.clip(img, 0, 255)
        
        return img
        
    def get_transformed_vecs(self):
        img_path = list()
        count_list = list()
        print('Adding noise')
        
        for i in range(861):
            img_path.append(os.path.join(self.data_path, self.horse_name_dset[i].decode()))
            count_list.append(self.horse_count_dset[i])
            
        for i in range(self.n_additional_random):
            idx = random.randint(0, 27428)
            while os.path.join(self.data_path, self.name_dset[idx].decode()) in img_path and self.count_dset[idx] > 0:
                idx = random.randint(0, 27428)
                
            img_path.append(os.path.join(self.data_path, self.name_dset[idx].decode()))
            count_list.append(self.count_dset[idx])
            
        print('Got image list')
        ret = list()
        for img in img_path:
            transformed = self.transform_img(img)
            ret.append(transformed)
            
        return self.net.forward_np_array(np.array(ret)), count_list
        