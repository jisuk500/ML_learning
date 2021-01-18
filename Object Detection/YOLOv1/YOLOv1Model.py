# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 00:47:25 2020

@author: jisuk
"""

#%% import modules
import os
import sys

import config as cfg

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, LeakyReLU

import VGG16_subclass as VGG
import Resnet50_subclass as ResNet

#%% classes

class YOLOv1Model(tf.keras.Model):
    def __init__(self):
        super(YOLOv1Model, self).__init__()
        
        if cfg.YOLOConfig['frontNet'] == "VGG16":
            self.front = tf.keras.Sequential(name='VGG16_front')
            self.front.add(VGG.VGG16(1000))
        elif cfg.YOLOConfig['frontNet'] == "ResNet50":
            self.front = tf.keras.Sequential(name='ResNet_front')
            self.front.add(ResNet.ResNet50(1000))
        
        
        self.conv1 = tf.keras.Sequential(name='conv1')
        self.conv1.add(tf.keras.layers.ZeroPadding2D(padding=(1,1)))
        self.conv1.add(Conv2D(1024,(3,3)))
        self.conv1.add(BatchNormalization())
        self.conv1.add(LeakyReLU(alpha=0.1))
        
        self.conv2 = tf.keras.Sequential(name='conv2')
        self.conv2.add(Conv2D(1024,(3,3),padding='same'))
        self.conv2.add(BatchNormalization())
        self.conv2.add(LeakyReLU(alpha=0.1))
        
        
        self.conv3 = tf.keras.Sequential(name='conv3')
        self.conv3.add(MaxPooling2D((2,2),strides=2))
        self.conv3.add(Conv2D(1024,(3,3),padding='same'))
        self.conv3.add(BatchNormalization())
        self.conv3.add(LeakyReLU(alpha=0.1))
        
        self.conv4 = tf.keras.Sequential(name='conv4')
        self.conv4.add(Conv2D(1024,(3,3),padding='same'))
        self.conv4.add(BatchNormalization())
        self.conv4.add(LeakyReLU(alpha=0.1))
        
        self.FC = tf.keras.Sequential(name='FC')
        self.FC.add(Flatten())
        self.FC.add(Dense(512))
        self.FC.add(Dense(4096))
        self.FC.add(LeakyReLU(alpha=0.1))
        
        self.outputLayer = tf.keras.Sequential(name='outputLayer')
        self.outputLayer.add(Dense(1470,activation='sigmoid'))
        
        
    def call(self, x, training=False):
        
        x = self.front(x)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.FC(x)
        
        x = self.outputLayer(x)
        
        return x
#%% examples

y = YOLOv1Model()
y.build((None,448,448,3))
y.summary()
        