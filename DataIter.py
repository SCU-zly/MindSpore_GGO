#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os,sys
import time
import cv2
from scipy.ndimage.interpolation import rotate
import pandas as pd

class GGODataIter(Dataset):
    def __init__(self, data_file, phase="train",crop_size=48,crop_depth=16,aug=1,sample_phase='over',classifier_type=0):
        # self.data_dir = data_dir 
        self.phase = phase
        # import pdb;pdb.set_trace()
        self.data_arr = np.load(data_file,allow_pickle=True).tolist()
        self.classifier_type = classifier_type
        
        AAH_lst = self.data_arr['AAH']
        AIS_lst = self.data_arr['AIS']
        MIA_lst = self.data_arr['MIA']
        IA_lst = self.data_arr['IA']
        self.AAH_lst = AAH_lst
        self.AIS_lst = AIS_lst
        self.MIA_lst = MIA_lst
        self.IIA_lst = IA_lst
        
        random.shuffle(AAH_lst)
        random.shuffle(AIS_lst)
        random.shuffle(MIA_lst)
        random.shuffle(IA_lst)
        
        self.NIA_lst = []
        self.MIA_lst = []
        self.IA_lst = []
                
        if self.classifier_type == 0:
            print("Modality 0")
            self.NIA_lst = AAH_lst + AIS_lst + MIA_lst
            self.IA_lst = IA_lst
            
            if phase == "train":
                minus_NIA = len(self.IA_lst) - len(self.NIA_lst)
                if sample_phase == 'over':
                    random.shuffle(self.NIA_lst)
                    NIA_cop =self. NIA_lst[:minus_NIA]
                    self.data_lst = self.IA_lst + self.NIA_lst + NIA_cop
                elif sample_phase == 'under':
                    random.shuffle(self.NIA_lst)
                    IA_cop = self.IA_lst[:len(self.NIA_lst)]
                    self.data_lst = IA_cop + self.NIA_lst
                else:
                    self.data_lst = self.IA_lst + self.NIA_lst
            else:
                random.shuffle(self.NIA_lst)
                IA_cop = self.IA_lst[:len(self.NIA_lst)]
                self.data_lst = self.IA_lst + self.NIA_lst
                
        elif self.classifier_type == 1:
            print("Modality 1")
            self.NIA_lst = AAH_lst + AIS_lst 
            self.MIA_lst = MIA_lst
            self.IA_lst = IA_lst
            
            if phase == "train":
                minus_MIA = len(self.IA_lst) - len(self.MIA_lst)
                num_nia = 1 # ·int(len(self.IA_lst) / len(self.NIA_lst))
                # import pdb;pdb.set_trace()
                if sample_phase == 'over':
                    random.shuffle(self.NIA_lst)
                    MIA_cop = self.MIA_lst[:minus_MIA]
                    self.data_lst = self.IA_lst + self.MIA_lst + MIA_cop + self.NIA_lst * num_nia
                    
                elif sample_phase == 'under':
                    random.shuffle(self.NIA_lst)
                    IA_cop = self.IA_lst[:len(self.MIA_lst)]
                    self.data_lst = IA_cop + self.NIA_lst + self.MIA_lst
                else:
                    self.data_lst = self.IA_lst + self.NIA_lst+ self.MIA_lst
            else:
                random.shuffle(self.MIA_lst)
                IA_cop = self.IA_lst[:len(self.MIA_lst)]
                self.data_lst = self.IA_lst + self.NIA_lst+ self.MIA_lst
            
        elif classifier_type == 2:
            print("Modality 2")
            self.NIA_lst = AAH_lst + AIS_lst 
            self.IA_lst = IA_lst + MIA_lst
            # import pdb;pdb.set_trace()
            if phase == "train":
                num_NIA = int(len(self.IA_lst) / len(self.NIA_lst))
                if sample_phase == 'over':
                    # random.shuffle(self.NIA_lst)
                    # NIA_cop = self.NIA_lst[:minus_NIA]
                    self.data_lst = self.IA_lst + self.NIA_lst * num_NIA
                elif sample_phase == 'under':
                    random.shuffle(self.NIA_lst)
                    IA_cop = self.IA_lst[:len(self.NIA_lst)]
                    self.data_lst = IA_cop + self.NIA_lst
                else :
                    random.shuffle(self.NIA_lst)
                    IA_cop = self.IA_lst[:len(self.NIA_lst)]
                    self.data_lst = self.IA_lst + self.NIA_lst
                    # import pdb;pdb.set_trace()
            else:
                random.shuffle(self.NIA_lst)
                IA_cop = IA_lst[:len(self.NIA_lst)]
                self.data_lst = self.IA_lst + self.NIA_lst
                # import pdb;pdb.set_trace()
        
        self.data_lst = self.data_lst * aug
        
        random.shuffle(self.data_lst)
        # import pdb;pdb.set_trace()
        print("The total samples is %d" % self.__len__())
        self.crop = Crop(size=crop_size,zslice=crop_depth,phase=self.phase)
        
    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t%1)[2:7]))#seed according to time
        
        if self.classifier_type == 0:
            cur_dir = self.data_lst[idx]
            label_lst = cur_dir.split('_')
            label = np.zeros((1,),dtype=np.float32)
            
            if cur_dir in self.IA_lst:
                label[0] = 1.0
            else:
                label[0] = 0.0

            if self.phase == "train":
                cur_idx = idx
            else:
                cur_idx = idx
            imgs = self.crop(cur_dir)
            # import pdb;pdb.set_trace()
            imgs = (imgs - 128) / 255.0
            return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
        
        elif self.classifier_type == 2:
            
            cur_dir = self.data_lst[idx]
            label_lst = cur_dir.split('_')
            label = np.zeros((1,),dtype=np.float32)
            if cur_dir in self.IA_lst:
                label[0] = 1.0
            else:
                label[0] = 0.0

            if self.phase == "train":
                cur_idx = idx
            else:
                cur_idx = idx
            imgs = self.crop(cur_dir)
            # import pdb;pdb.set_trace()
            imgs = (imgs - 128) / 255.0
            return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
        
        else:
            cur_dir = self.data_lst[idx]
            label_lst = cur_dir.split('_')
            label = np.zeros((3,),dtype=np.float32)
            if cur_dir in self.IA_lst:
                label[2] = 1.0
            elif cur_dir in self.MIA_lst:
                label[1] = 1.0
            else:
                label[0] = 1.0

            if self.phase == "train":
                cur_idx = idx
            else:
                cur_idx = idx
            imgs = self.crop(cur_dir)
            # import pdb;pdb.set_trace()
            imgs = (imgs - 128) / 255.0
            return torch.from_numpy(imgs.astype(np.float32)), torch.from_numpy(label.astype(np.float32)),cur_dir
            
    
    def normlize(self,img):
        MIN_BOUND = -1400
        MAX_BOUND = 400
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img[img>1] = 1
        img[img<0] = 0
        return img
    
    def  __len__(self):
        if self.phase == 'train':
            return len(self.data_lst)
        elif self.phase =='test':
            return len(self.data_lst)
        else:
            return len(self.sample_bboxes)
        
class CenterCrop(object):
    def __init__(self, size, zslice):
        assert size in [16,32,48,64,96] and zslice in [6,8,10,16]
        self.size = (int(size), int(size))
        self.zslice = zslice

    def __call__(self,data):
        s, y, x = data.shape
        des_w, des_h = self.size
        des_s = self.zslice
        x_start = max(int(round((x - des_w) / 2.)),0)
        x_end = min(x_start+des_w,x)
        
        y_start = max(int(round((y - des_h) / 2.)),0)
        y_end = min(y_start+des_h, y)
        
        s_start = max(int(round((s - des_s) / 2.)),0)
        s_end = min(s_start+des_s,s)
        
        data = data[s_start : s_end,
                    y_start : y_end,
                    x_start : x_end]
        
        pad_size = (des_s-(s_end-s_start), des_h-(y_end-y_start), des_w-(x_end-x_start))
        pad_edge = ((int(pad_size[0]/2),pad_size[0] - int(pad_size[0]/2)),(int(pad_size[1]/2),pad_size[1] - int(pad_size[1]/2)),(int(pad_size[2]/2),pad_size[2] - int(pad_size[2]/2)))
        
        if np.sum(pad_size) != 0:
            data = np.pad(data, pad_edge, 'edge')
            
        try:
            data = data.reshape(des_s,des_h,des_w)
        except:
            import pdb;pdb.set_trace()
        return data

class RandomCenterCrop(object):
    def __init__(self, size, zslice):
        # import pdb;pdb.set_trace()
        assert size in [16,32,48,64,96] and zslice in [6,8,10,16]
        self.size = (int(size), int(size))
        self.zslice = zslice
        if size == 16:
            self.randseed = 4
        elif size == 32:
            self.randseed = 6
        elif size == 48:
            self.randseed = 8
        elif size == 64:
            self.randseed = 10
        elif size == 96:
            self.randseed = 12
            
    def __call__(self, data):
        s, y, x = data.shape
        des_w, des_h = self.size
        des_s = self.zslice
        
        i = random.randint(-self.randseed, self.randseed)
        j = random.randint(-self.randseed, self.randseed)
        
        x_start = max(int(round((x - des_w) / 2.) + i),0)
        x_end = min(x_start+des_w,x)
        
        y_start = max(int(round((y - des_h) / 2.) + j),0)
        y_end = min(y_start+des_h, y)
        
        s_start = max(int(round((s - des_s) / 2.)),0)
        s_end = min(s_start+des_s,s)
        
        data = data[s_start : s_start + des_s,
                    y_start : y_start + des_h,
                    x_start : x_start + des_w]
        
        pad_size = (des_s-(s_end-s_start), des_h-(y_end-y_start), des_w-(x_end-x_start))
        pad_edge = ((int(pad_size[0]/2),pad_size[0] - int(pad_size[0]/2)),(int(pad_size[1]/2),pad_size[1] - int(pad_size[1]/2)),(int(pad_size[2]/2),pad_size[2] - int(pad_size[2]/2)))
        
        if np.sum(pad_size) != 0:
            data = np.pad(data, pad_edge, 'edge')
            
        data = data.reshape(des_s,des_h,des_w)
        return data
        
class Crop(object):
    def __init__(self,size=48,zslice=16,phase='train'):
        self.crop_size = size
        self.zslice = zslice
        self.phase = phase
        self.random_crop = RandomCenterCrop(size,zslice)
        self.center_crop = CenterCrop(size,zslice)
    
    # normlization of CT values 
    def normlize(self,img):
        MIN_BOUND = -1200
        MAX_BOUND = 0
        img = (img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img[img>1] = 1
        img[img<0] = 0
        return img
    
    def __call__(self,img_npy):
        img = np.load(img_npy)
        shape = img.shape
        for shape_ in shape:
            if shape_ == 0:
                import pdb;pdb.set_trace()
        if self.phase == "test":
            img_r = self.center_crop(img)
        else:
            img_r = self.random_crop(img)
            
        if self.phase == "train":
            # import pdb;pdb.set_trace()
            ran_type = random.randint(0,1)
            if ran_type == 0:
                angle1 = np.random.rand()*180
                img_r = rotate(img_r,angle1,axes=(1,2),reshape=False)
            elif ran_type == 1:
                angle1 = np.random.rand()*180
                img_r = rotate(img_r,angle1,axes=(1,2),reshape=False)
            
        for shapa_ in img_r.shape[1:]:
            if shapa_ not in [16,32,48,64,96]:
                print(shapa_)
                import pdb;pdb.set_trace()
        return img_r[np.newaxis,:,:,:]
    
    mal_lst = []
    
    