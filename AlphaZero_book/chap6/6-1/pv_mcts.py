# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 23:41:28 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%% 패키지 임포트
from game import State
from duel_network import DN_INPUT_SHAPE
from math import sqrt
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np

#%% 파라미터 준비
PV_EVALUATE_COUNT = 50 # 추론 1회당 시뮬레이션 횟수(오리지널: 1600회)

#%% 추론
def predict(model, state):
    # 추론을 위한 입력 데이터 변환
    a,b,c, = DN_INPUT_SHAPE
    x = np.array([state.pieces, state.enemy_pieces])
    x = x.reshape(c,a,b).transpose(1,2,0).reshape(1,a,b,c)
    
    # 추론
    y = model.predict(x,batch_size=1)
    
    # 정책 얻기
    policies = y[0][0][list(state.legal_actions())]
    policies /= sum(policies) if sum(policies) else 1
    
    # 가치 얻기
    value = y[1][0][0]
    return policies,value

#%% 노드의 리스트를 시행 횟수 리스트로 반환
def nodes_to_scores(nodes):
    scores = []
    for c in nodes:
        scores.append(c.n)
    return scores

#%% 
