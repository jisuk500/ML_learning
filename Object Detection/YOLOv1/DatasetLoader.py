# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 13:01:33 2020

@author: jisuk
"""

#%% import module
import os 
import sys
import numpy as np
import random

import config as cfg
import cv2



#%% class

class DatasetLoader:
    def __init__(self, dataTextPath, isTraining=True):
        self.imageRoot = os.path.join(cfg.VOCdatasetConfig['rootDir'],cfg.VOCdatasetConfig['imageFolder'])
        self.dataTextPath = dataTextPath
        
        self.inputSize = cfg.YOLOConfig['inputSize']
        self.outputGrid = cfg.YOLOConfig['outputGrid']
        self.batchSize = cfg.YOLOConfig['batchSize']
        
        self.train = isTraining
        
        self.fnames = []
        self.bboxes = []
        self.labels = []
        
        self.load_fnames_bboxes()
        
        self.record_point = 0
        self.total_samples = len(self.fnames)
        self.num_batch_per_epoch = int(self.total_samples / self.batchSize)
        
        
    def load_fnames_bboxes(self):
        fs_input = open(self.dataTextPath,'r')
        
        for line in fs_input.readlines():
            line = line.strip().split()
            self.fnames.append(os.path.join(self.imageRoot,line[0]))
            num_bboxes = (len(line) - 1) // 5
            box = []
            label = []
            
            for i in range(num_bboxes):
                x = float(line[1 + 5*i])
                y = float(line[2 + 5*i])
                x2 = float(line[3 + 5*i])
                y2 = float(line[4 + 5*i])
                
                c = line[5 + 5*i]
                
                box.append([x, y, x2, y2])
                label.append(c)
            
            self.bboxes.append(box)
            self.labels.append(label)
            
        fs_input.close()
            
            
    def parseData(self,idx):
        fname = self.fnames[idx]
        img = cv2.imread(fname)
        boxes = np.array(self.bboxes[idx]).copy()
        labels = np.array(self.labels[idx]).copy()
        
        if self.train:
            img, boxes = self.random_Flip(img, boxes)
            img, boxes = self.random_Scale(img, boxes)
            img = self.random_Blur(img, boxes)
            img = self.random_Brightness(img, boxes)
            img = self.random_Hue(img, boxes)
            img = self.random_Saturation(img, boxes)
            img, boxes, labels = self.random_Shift(img, boxes, labels)
            img, boxes, labels = self.random_Crop(img, boxes, labels)
        
        h,w,_ = img.shape
        boxes = boxes / [w,h,w,h]
        img = cv2.resize(img,(self.inputSize[0],self.inputSize[1]))
        
        target = self.encoder(boxes,labels)
        return img,target
        
            
    
    def random_Flip(self, img, boxes):
        if random.random() < 0.5:
	        im_lr = np.fliplr(img).copy()
	        h,w,_ = img.shape
	        xmin = w - boxes[:,2]
	        xmax = w - boxes[:,0]
	        boxes[:,0] = xmin
	        boxes[:,2] = xmax
	        return im_lr, boxes
        return img, boxes
    
    def random_Scale(self, img, boxes):
        if random.random() < 0.5:
	        scale = random.uniform(0.8,1.2)
	        height,width,c = img.shape
	        img = cv2.resize(img,(int(width*scale),height))
	        scale_boxes = [scale,1,scale,1]
	        boxes = boxes * scale_boxes
	        return img,boxes
        return img,boxes
    
    def random_Blur(self, img, boxes):
        if random.random()<0.5:
	        img = cv2.blur(img,(5,5))
        return img
    
    def random_Brightness(self, img, boxes):
        if random.random() < 0.5:
	        hsv = self.BGR2HSV(img)
	        h,s,v = cv2.split(hsv)
	        adjust = random.choice([0.5,1.5])
	        v = v*adjust
	        v = np.clip(v, 0, 255).astype(hsv.dtype)
	        hsv = cv2.merge((h,s,v))
	        img = self.HSV2BGR(hsv)
        return img
    
    def random_Hue(self, img, boxes):
        if random.random() < 0.5:
	        hsv = self.BGR2HSV(img)
	        h,s,v = cv2.split(hsv)
	        adjust = random.choice([0.5,1.5])
	        h = h*adjust
	        h = np.clip(h, 0, 255).astype(hsv.dtype)
	        hsv = cv2.merge((h,s,v))
	        img = self.HSV2BGR(hsv)
        return img
    
    def random_Saturation(self, img, boxes):
        if random.random() < 0.5:
	        hsv = self.BGR2HSV(img)
	        h,s,v = cv2.split(hsv)
	        adjust = random.choice([0.5,1.5])
	        s = s*adjust
	        s = np.clip(s, 0, 255).astype(hsv.dtype)
	        hsv = cv2.merge((h,s,v))
	        img = self.HSV2BGR(hsv)
        return img
    
    def random_Shift(self, img, boxes, labels):
        center = (boxes[:,2:]+boxes[:,:2])/2
        if random.random() <0.5:
	        height,width,c = img.shape
	        after_shfit_image = np.zeros((height,width,c),dtype=img.dtype)
	        after_shfit_image[:,:,:] = (104,117,123) #bgr
	        shift_x = random.uniform(-width*0.2,width*0.2)
	        shift_y = random.uniform(-height*0.2,height*0.2)
	        #print(bgr.shape,shift_x,shift_y)
	        if shift_x>=0 and shift_y>=0:
	            after_shfit_image[int(shift_y):,int(shift_x):,:] = img[:height-int(shift_y),:width-int(shift_x),:]
	        elif shift_x>=0 and shift_y<0:
	            after_shfit_image[:height+int(shift_y),int(shift_x):,:] = img[-int(shift_y):,:width-int(shift_x),:]
	        elif shift_x <0 and shift_y >=0:
	            after_shfit_image[int(shift_y):,:width+int(shift_x),:] = img[:height-int(shift_y),-int(shift_x):,:]
	        elif shift_x<0 and shift_y<0:
	            after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = img[-int(shift_y):,-int(shift_x):,:]

	        shift_xy = [int(shift_x),int(shift_y)]
	        center = center + shift_xy
	        mask1 = np.where((center[:,0]>0) & (center[:,0]<width))[0]
	        mask2 = np.where((center[:,1]>0) & (center[:,1]<height))[0]
	        mask = np.intersect1d(mask1,mask2)
	        boxes_in = boxes[mask]
	        if len(boxes_in) == 0:
	            return img,boxes,labels
	        box_shift = [int(shift_x),int(shift_y),int(shift_x),int(shift_y)]
	        boxes_in = boxes_in+box_shift
	        labels_in = labels[mask]
	        return after_shfit_image,boxes_in,labels_in
        return img,boxes,labels
    
    def random_Crop(self, img, boxes, labels):
        if random.random() < 0.5:
	        center = (boxes[:,2:]+boxes[:,:2])/2
	        height,width,c = img.shape
	        h = random.uniform(0.6*height,height)
	        w = random.uniform(0.6*width,width)
	        x = random.uniform(0,width-w)
	        y = random.uniform(0,height-h)
	        x,y,h,w = int(x),int(y),int(h),int(w)

	        center = center - [x,y]
	        mask1 = np.where((center[:,0]>0) & (center[:,0]<w))[0]
	        mask2 = np.where((center[:,1]>0) & (center[:,1]<h))[0]
	        mask = np.intersect1d(mask1,mask2)

	        boxes_in = boxes[mask]
	        if(len(boxes_in)==0):
	            return img,boxes,labels
	        box_shift = [x,y,x,y]

	        boxes_in = boxes_in - box_shift
	        boxes_in[:,0]=boxes_in[:,0].clip(min=0,max=w)
	        boxes_in[:,2]=boxes_in[:,2].clip(min=0,max=w)
	        boxes_in[:,1]=boxes_in[:,1].clip(min=0,max=h)
	        boxes_in[:,3]=boxes_in[:,3].clip(min=0,max=h)

	        labels_in = labels[mask]
	        img_croped = img[y:y+h,x:x+w,:]
	        return img_croped,boxes_in,labels_in
        return img,boxes,labels
        
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    def BGR2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    def HSV2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2BGR)

    def RGB2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
    def encoder(self, boxes, labels):
        target = np.zeros((self.outputGrid[0],self.outputGrid[1],30))
        cell_size = 1./self.outputGrid[0]
        wh = (boxes[:,2:] - boxes[:,0:2])
        cxcy = (boxes[:,2:] + boxes[:,0:2]) / 2
        
        for i in range(cxcy.shape[0]):
            _cxcy = cxcy[i]
            ij = (np.ceil(_cxcy/cell_size)-1).astype(np.int32)
            target[ij[1]][ij[0],4] = 1
            target[ij[1]][ij[0],9] = 1
            target[ij[1],ij[0],int(labels[i])+9] = 1
            xy = ij*cell_size
            delta_xy = (_cxcy-xy)/cell_size
            target[ij[1],ij[0],2:4] = wh[i]
            target[ij[1],ij[0],0:2] = delta_xy
            target[ij[1],ij[0],7:9] = wh[i]
            target[ij[1],ij[0],5:7] = delta_xy
        
        return target
    
    
    def batch(self):
        if self.record_point % self.num_batch_per_epoch == 0:
            self.shuffle_idx = np.random.permutation(self.total_samples) if self.train else np.arange(self.total_samples)
            self.record_point = 0

        images,targets = [],[]
        idxs = self.shuffle_idx[self.record_point*self.batchSize:(self.record_point+1)*self.batchSize]
        for idx  in idxs:
            image,target = self.parseData(idx)
            images.append(image)
            targets.append(target)
        
        images = np.asarray(images, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)

        self.record_point+=1
        return images,targets

#%% test

# a = DatasetLoader(os.path.join(cfg.VOCdatasetConfig['textFileFolder'],"train.txt"),True)
# a = DatasetLoader(os.path.join(cfg.VOCdatasetConfig['textFileFolder'],"val.txt"),True)
