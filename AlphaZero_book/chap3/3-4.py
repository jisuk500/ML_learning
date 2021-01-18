# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:53:45 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%

# 3-4-3 패키지 임포트

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks  import LearningRateScheduler
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

# 데이터 세트 준비
(train_images, train_labels) , (test_images, test_labels) = cifar10.load_data()

# 데이터 세트 전처리
train_images = train_images
train_labels = to_categorical(train_labels, 10)
test_images = test_images
test_labels = to_categorical(test_labels,10)

# 데이터 세트 전처리 후 형태 확인
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

#%%

# 3-4-6 functional api 이용

# 모델 생성

#컨볼루셔널 레이어 생성



def conv(filters, kernel_size, stride=1):
    return Conv2D(filters, kernel_size, strides=(stride,stride), padding='same', 
                  use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(0.0001))

# 레지듀얼 블록 A 생성
def first_residual_unit(filters, strides):
    def f(x):
        # BN -> ReLU
        x = BatchNormalization()(x)
        b = Activation('relu')(x)

        # 컨볼루셔널 레이어
        x = conv(filters // 4, 1, strides)(b)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 컨볼루셔널 레이어 -> BN -> ReLU
        x = conv(filters // 4 , 3)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 컨볼루셔널 레이어 ->
        x = conv(filters,1)(x)

        # 숏컷 형태 사이즈 조정
        sc = conv(filters, 1, strides)(b)

        # add
        return Add()([x, sc])

    return f

# 레지듀얼 블록 B 생성
def residual_unit(filters):
    def f(x):
        sc = x
        # -> BN -> ReLU
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 컨볼루셔널 레이어 -> BN -> ReLU
        x = conv(filters // 4, 1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 컨볼루셔널 레이어->
        x = conv(filters,1)(x)

        # add
        return Add()([x, sc])
    return f

# 레지듀얼 블록 A, 레지듀얼 블록 B x 17
def residual_block(filters, strides, unit_size):
    def f(x):
        x = first_residual_unit(filters, strides)(x)
        for i in range(unit_size - 1):
            x = residual_unit(filters)(x)
        return x
    return f


# 입력 데이터 형태
input = Input(shape=(32,32,3))

# 컨볼루셔널 레이어
x = conv(16,3)(input)

# 레지듀얼 블록 x 54
x = residual_block(64,1,18)(x)
x = residual_block(128,2,18)(x)
x = residual_block(256,2,18)(x)

# BN -> ReLU
x = BatchNormalization()(x)
x = Activation('relu')(x)

# 풀링 레이어
x = GlobalAveragePooling2D()(x)

# 전결합 레이어
output = Dense(10,activation='softmax', kernel_regularizer=l2(0.0001))(x)

#모델 생성
model = Model(inputs=input, outputs = output)

# 3-4-8 컴파일
# 컴파일
model.compile(loss='categorical_crossentropy',optimizer=SGD(momentum=0.9), metrics=['acc'])

#%%

# ImageDataGenerator 준비

train_gen = ImageDataGenerator(
    featurewise_center=True,featurewise_std_normalization=True,
    width_shift_range = 0.125, height_shift_range=0.125,
    horizontal_flip=True)
test_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

for data in (train_gen, test_gen):
    data.fit(train_images)

# LearningRateSchedular 준비
def step_decay(epoch):
    x = 0.1
    if epoch >= 80: x = 0.01
    if epoch >= 120: x = 0.001
    return x

lr_decay_cb = LearningRateScheduler(step_decay)

#%%

# 3-4-11 학습
batch_size = 128
history = model.fit(
    train_gen.flow(
        train_images,
        train_labels,
        batch_size=batch_size),
    epochs=200,
    steps_per_epoch=train_images.shape[0] // batch_size,
    validation_data = test_gen.flow(
        test_images,
        test_labels,
        batch_size = batch_size),
    validation_steps = test_images.shape[0] // batch_size,
    callbacks=[lr_decay_cb]
)

#%%

# 3-4-12 모델 저장
model.save("3-4-resnet.h5")

#%%

# 3-4-13 그래프 표시
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()


#%%

# 3-4-14 평가

batch_size = 128
test_loss , test_acc = model.evaluate_generator(
    test_gen.flow(test_images, test_labels, batch_size=batch_size),
    steps=10
)

print("loss: {:.3f}\nacc: {:.3f}".format(test_loss, test_acc))