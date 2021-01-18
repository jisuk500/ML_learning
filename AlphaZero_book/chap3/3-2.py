# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:53:38 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%

# 3-2-3 패키지 임포트
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#데이터 세트 준비
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# 데이터 세트 형태 확인
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

# 데이터 세트의 데이터 확인
column_names = ["CRIN","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
df = pd.DataFrame(train_data, columns=column_names)
df.head()

# 데이터 세트의 라벨 확인
#print(train_labels[0:10])

#%%

# 3-2-5 데이터 세트 전처리 및 확인

# 데이터 세트 전처리(셔플)
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# 데이터 세트 전처리 (표준화)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
mean = test_data.mean(axis=0)
std = test_data.std(axis=0)
test_data = (test_data - mean) / std

# 데이터 세트 전처리 후의 데이터 확인
column_names = ["CRIN","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
df = pd.DataFrame(train_data, columns=column_names)
df.head()

#%%

# 3-2-6 모델 생성

# 모델 생성
model = Sequential()
model.add(Dense(64, activation='relu',input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

# 3-2-7 컴파일

# 컴파일
model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['mean_absolute_error'])

#%%

# 3-2-8 학습

# EarlyStopping 준비
early_stop = EarlyStopping(monitor='val_loss', patience=50)

# 학습
history = model.fit(train_data, train_labels, epochs=1000, validation_split=0.2, callbacks=[early_stop])

#%%

# 3-2-9 그래프 표시
print(history.history)

plt.plot(history.history['mean_absolute_error'], label='train mae')
plt.plot(history.history['val_mean_absolute_error'], label='val mae')
plt.xlabel('epoch')
plt.ylabel('mae [1000$]')
plt.legend(loc = 'best')
plt.ylim([0,5])
plt.show()

#%%

# 3-2-10 평가

# 평가
test_loss , test_mae = model.evaluate(test_data, test_labels)
print('loss: {:.3f}\nname: {:.3f}'.format(test_loss, test_mae))

# 실제 가격 표시
print(np.round(test_labels[0:10]))

# 추론한 가격 표시
test_predictions = model.predict(test_data[0:10]).flatten()
print(np.round(test_predictions))


