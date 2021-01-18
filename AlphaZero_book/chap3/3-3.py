# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:53:41 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%

# 3-3-4 패키지 임포트

# 패키지 임포트
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 데이터 세트 준비
(train_images, train_labels) , (test_images, test_labels) = cifar10.load_data()

# 데이터 세트 형태 확인
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

# 데이터 세트 이미지 확인
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(train_images[i])
plt.show()

# 데이터 세트 라벨 확인
print(train_labels[0:10])


#%%

# 3-3-6 데이터 세트 전처리 및 확인

# 데이터 세트 이미지 전처리
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# 데이터 세트 이미지 전처리 후 형태 확인
print(train_images.shape)
print(test_images.shape)

# 데이터 세트 라벨 전처리
train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels,10)

# 데이터 세트 라벨 전처리 후 형태 확인
print(train_labels.shape)
print(test_labels.shape)

#%%

# 3-3-7 모델 생성

# 모델 생성
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Dropout(0.25))

# Conv->Conv -> Pool -> Dropout
model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
model.add(Conv2D(64,(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))

# Flatten->Dense->Dropout->Dense
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 3-3-8 컴파일

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

# 3-3-9 학습
# 학습
history = model.fit(train_images, train_labels, batch_size = 500, epochs=30, validation_split=0.1)

#%%

# 3-3-10 모델 저장과 로드

model.save('3-3-convolution.h5')

# 그래프 표시
plt.plot(history.history['acc'],label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

#%%

# 3-3-12 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc))

# 3-3-13 추론
# 추론할 이미지 표시

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(test_images[i])
plt.show()

# 추론한 라벨 표시
import random 
n = random.sample(range(1,10001),10)
test_predictions = model.predict(test_images[n])
test_predictions = np.argmax(test_predictions, axis=1)
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print([labels[n] for n in test_predictions])
temp = np.argmax(test_labels[n],axis=1)
print([labels[n] for n in temp])

