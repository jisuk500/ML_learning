# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 01:07:06 2020

@author: https://github.com/eremo2002/tf.keras-CNN/blob/master/vgg16_subclass.py
"""

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense

class VGG16(tf.keras.Model):
    def __init__(self, nb_classes):
        super(VGG16, self).__init__()

        self.nb_class = nb_classes
        
        self.stage1 = tf.keras.Sequential(name="stage1")
        self.stage1.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.stage1.add(BatchNormalization())
        self.stage1.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.stage1.add(BatchNormalization())
        self.stage1.add(MaxPooling2D((2, 2), strides=(2, 2)))
                        
                        
                        
        self.stage2 = tf.keras.Sequential(name="stage2")
        self.stage2.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.stage2.add(BatchNormalization())
        self.stage2.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.stage2.add(BatchNormalization())
        self.stage2.add(MaxPooling2D((2, 2), strides=(2, 2)))
                        
        self.stage3 = tf.keras.Sequential(name="stage3")
        self.stage3.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.stage3.add(BatchNormalization())
        self.stage3.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.stage3.add(BatchNormalization())
        self.stage3.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.stage3.add(BatchNormalization())
        self.stage3.add(MaxPooling2D((2, 2), strides=(2, 2)))
                        
        self.stage4 = tf.keras.Sequential(name="stage4")
        self.stage4.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.stage4.add(BatchNormalization())
        self.stage4.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.stage4.add(BatchNormalization())
        self.stage4.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.stage4.add(BatchNormalization())
        self.stage4.add(MaxPooling2D((2, 2), strides=(2, 2)))
                        
        self.stage5 = tf.keras.Sequential(name="stage5")
        self.stage5.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.stage5.add(BatchNormalization())
        self.stage5.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.stage5.add(BatchNormalization())
        self.stage5.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        self.stage5.add(BatchNormalization())
        self.stage5.add(MaxPooling2D((2, 2), strides=(2, 2)))
                      
                        
        # self.dense = tf.kears.sequential()
        # self.dense.add(Flatten())
        # self.dense.add(Dense(4096,activation='relu'))
        # self.dense.add(Dense(4096,activation='relu'))
        # self.dense.add(Dense(nb_classes,activation='relu'))

    def call(self, x, training=False):
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        # x = self.dense(x)

        return x
    


model = VGG16(1000)
model.build(input_shape=(None,448,448,3))
model.summary()