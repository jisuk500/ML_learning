# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:04:03 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""
#%% 모듈 임포트

from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os

#%% 파라미터 준비
DN_FILTERS = 128 # 컨볼루셔널 레이어 커널 수(오리지널 256)
DN_RESIDUAL_NUM = 16 # 레지듀얼 블록 수 (오리지널 19)
DN_INPUT_SHAPE = (3,3,2) # 입력 형태
DN_OUTPUT_SIZE = 9 # 행동 수(배치수*(3x3))

#%% 컨볼루셔널 레이어 생성

def conv(filters):
    return Conv2D(filters,3,padding='same', use_bias=False,
                  kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))

#%% 레지듀얼 블록 생성

def residual_block():
    def f(x):
        sc = x
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = conv(DN_FILTERS)(x)
        x = BatchNormalization()(x)
        x = Add()([x,sc])
        x = Activation('relu')(x)
        return x
    return f

#%% 듀얼 네트워크 생성

def dual_network():
    # 모델 생성이 완료된 경우, 처리하지 않음
    if os.path.exists('best.h5'):
        return
    
    # 입력 레이어
    input = Input(shape=DN_INPUT_SHAPE)
    
    # 컨볼루셔널 레이어
    x = conv(DN_FILTERS)(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 레지듀얼 블록 16개
    for i in range(DN_RESIDUAL_NUM):
        x = residual_block()(x)
    
    # 풀링 레이어
    x = GlobalAveragePooling2D()(x)
    
    # 정책 출력
    p = Dense(DN_OUTPUT_SIZE, kernel_regularizer=l2(0.0005),
              activation='softmax', name='pi')(x)
    
    # 가치 출력
    v = Dense(1, kernel_regularizer=l2(0.0005))(x)
    v = Activation('tanh',name='v')(v)
    
    # 모델 생성
    model = Model(inputs=input, outputs=[p,v])
    
    # 모델 저장
    model.save('best.h5') # 베스트 플레이어 모델
    
    return model
    # # 모델 삭제
    # K.clear_session()
    # del model
    
#%% 동작 확인

if __name__ == "__main__":
    m = dual_network()
    
    