# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:02:01 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%

# 4-2-3 패키지 임포트

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


#%%

# 4-2-4 미로 생성 및 시각화

# 미로 생성
fig = plt.figure(figsize=(3,3))

# 벽
plt.plot([0,3],[3,3],color='k')
plt.plot([0,3],[0,0],color='k')
plt.plot([0,0],[0,2],color='k')
plt.plot([3,3],[1,3],color='k')
plt.plot([1,1],[1,2],color='k')
plt.plot([2,3],[2,2],color='k')
plt.plot([2,1],[1,1],color='k')
plt.plot([2,2],[0,1],color='k')

# 숫자
for i in range(3):
    for j in range(3):
        plt.text(0.5 + i, 2.5- j, str(i+j*3), size = 20, ha='center', va='center')

# 원
circle, = plt.plot([0.5], [2.5], marker='o', color='#d3d3d3', markersize=40)

# 눈금 및 배경 숨김
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=False, right=False, left=False, labelleft=False)
plt.box(False)

#%%

# 파라미터 theta 초기값 준비

theta_0 = np.array(
    [
     [np.nan, 1, 1, np.nan], # 0 상우하좌
     [np.nan, 1, 1, 1], # 1
     [np.nan, np.nan, np.nan, 1], # 2
     [1, np.nan, 1, np.nan], # 3
     [1, 1, np.nan, np.nan], # 4
     [np.nan, np.nan, 1, 1], # 5
     [1, 1, np.nan, np.nan], # 6
     [np.nan, np.nan, np.nan, 1], # 7
    ]
)

# 파라미터  theta 정책값으로 변환, 소프트맥스 사용

def get_pi(theta):
    #소프트맥스로 변
    [m,n] = theta.shape
    pi = np.zeros((m,n))
    exp_theta = np.exp(theta)

    for i in range(0,m):
        pi[i,:] = exp_theta[i,:] / np.nansum(exp_theta[i,:])

    pi = np.nan_to_num(pi)
    return pi

pi_0 = get_pi(theta_0)
print(pi_0)

#%%

# 4-2-8 정책에 따라 행동 얻기

def get_a(pi,s):
    return np.random.choice([0,1,2,3],p=pi[s])

# 행동에 따라 다음 상태 얻기

def get_s_next(s,a):
    if a==0 : # 상
        return s -3
    elif a==1:
        return s +1
    elif a==2:
        return s +3
    elif a==3:
        return s -1

# 에피소드 실행

def play(pi):
    s = 0 # 상태
    s_a_history = [[0,np.nan]] # 상태와 행동 이력

    # 에피소드 종료시까지 반복
    while True:
        # 정책에 따라 행동 얻기
        a = get_a(pi,s)

        #행동에 따라 다음 상태 얻기
        s_next = get_s_next(s,a)

        # 이력 갱신
        s_a_history[-1][1] = a
        s_a_history.append([s_next, np.nan])

        # 종료 판정
        if s_next == 8:
            break
        else:
            s = s_next
    return s_a_history

#에피소드 실행 및 이력 확인
s_a_history = play(pi_0)
print(s_a_history)
print('1 에피소드의 스텝 수: {}'.format(len(s_a_history)+1))

#%%

# 4-2-12 파라미터 theta 갱신

# 파라미터 theta 갱신
def update_theta(theta, pi, s_a_history):
    eta = 0.1 # 학습 계수
    total = len(s_a_history) - 1 # 골인 지점까지 걸린 총 스텝 수
    [s_count, a_count] = theta.shape

    # 파라미터 theta의 변화량 계산
    delta_theta = theta.copy()
    for i in range(0, s_count):
        for j in range(0,a_count):
            if not(np.isnan(theta[i,j])):
                #상태 s에서 행동 a를 선택한 횟수
                sa_ij = [sa for sa in s_a_history if sa == [i,j]]
                n_ij = len(sa_ij)

                # 상태 s에서 무엇인가 행동을 선택한 횟수
                sa_i = [sa for sa in s_a_history if sa[0] == i]
                n_i = len(sa_i)

                # 파라미터 theta의 변화량
                delta_theta[i,j] = (n_ij - pi[i,j] * n_i) / total
    
    # 파라미터 theta 갱신
    return theta + eta * delta_theta

#%%

# 4-2-15 반복하며 학습하여 실행하기

stop_epsilon = 10 ** -4 # 역치
theta = theta_0 # 파라미터 theta
pi = pi_0 # 정책

# 에피소드를 반복해서 실행하며 학습
for episodes in range(10000):
    # 1 에피소드 실행 후 이력 얻기
    s_a_history = play(pi)

    # 파라미터 theta 갱신
    theta = update_theta(theta, pi, s_a_history)

    # 정책 갱신
    pi_new = get_pi(theta)

    # 정책 변화량
    pi_delta = np.sum(np.abs(pi_new - pi))
    pi = pi_new

    # 출력
    print("에피소드: {}, 스텝: {}, 정책 변화량: {:.4f}".format(episodes, len(s_a_history) - 1, pi_delta))

    # 완료 판정
    if pi_delta < stop_epsilon: # 정책 변화량이 임곗값 이하
        break
    
#%%

# 4-2-15 애니메이션 표시

# 애니메이션 정기 처리를 수행하는 함수
def animate(i):
    state = s_a_history[i][0]
    circle.set_data((state%3) + 0.5,2.5 - int(state/3))
    return circle

# 애니메이션 표시
anim = animation.FuncAnimation(fig, animate, 
                               frames = len(s_a_history), interval=200, repeat=False)
HTML(anim.to_jshtml())

#%%

