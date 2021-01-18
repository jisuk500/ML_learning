# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:53:34 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%

# 3-1-3 패키지 임포트
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 3-1-4 데이터 준비 확인
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(train_images[i],'gray')

plt.show()

print(train_labels[0:10])

# 3-1-5 데이터 세트 전처리 및 확인
# 데이터 세트 이미지 전처리
train_images = train_images.reshape((train_images.shape[0],784))
test_images = test_images.reshape((test_images.shape[0],784))

# 데이터 세트 이미지 전처리 후 차원 확인
print(train_images.shape)
print(test_images.shape)

# 데이터 세트 라벨 전처리
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 데이터 세트 라벨 전처리 후 형태 확인
print(train_labels.shape)
print(test_labels.shape)

#%%

# 3-1-6 모델 생성
# 뉴럴 네트워크 모델을 생성한다.
model = Sequential()
model.add(Dense(256,activation='sigmoid',input_shape=(None,784))) # 입력 레이어
model.add(Dense(128,activation='sigmoid')) # 히든 레이어
model.add(Dropout(rate=0.5)) # 드롭 아웃
model.add(Dense(10,activation='softmax')) # 출력 레이어

model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.1),metrics=['acc'])

history = model.fit(train_images, train_labels, batch_size = 500, 
                    epochs=15, validation_split=0.2)

plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label='val_acc')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.show()

#%%

# 3-1-10 평가

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('loss: {:.3f}\nacc: {:.3f}'.format(test_loss, test_acc))

#%%

# 3-1-11 추론
import random

n = random.sample(range(10000),10)

#추론할 이미지 표시
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(test_images[n[i]].reshape((28,28)),'gray')
plt.show()

# 추론할 라벨 표시
test_predictions = model.predict(test_images[n])
test_predictions = np.argmax(test_predictions, axis=1)
print(test_predictions)
print(np.argmax(test_labels[n],axis=1))

