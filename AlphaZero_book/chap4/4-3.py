# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:02:10 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%


# 4-3-6 패키지 임포트

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML


#%%

# 4-3-7 미로 생성

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

# 4-3-8 랜덤 행동 준비

# 파라미터  theta 의 초기값 준비

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

# 4-3-10 행동 가치 함수 준비

[a,b] = theta_0.shape
Q = np.random.rand(a,b) * theta_0
# Q = 0 * theta_0
print(Q)

#%%

# 랜덤 또는 행동 가치 함수에 따라 행동 얻기

def get_a(s, Q, epsilon, pi_0):
    if np.random.rand() < epsilon:
        # 랜덤으로 행동 선택
        return np.random.choice([0,1,2,3],p=pi_0[s])
        
    else:
        # 행동 가치 함수로 행동 선택
        return np.nanargmax(Q[s])

# 행동에 따라 다음 상태 얻기
def get_s_next(s,a):
    if a == 0: # 상
        return s - 3
    elif a == 1: #우
        return s + 1
    elif a == 2: # 하
        return s + 3
    elif a == 3: # 좌
        return s - 1

# sarsa에 따른 행동 가치 함수 갱신

def sarsa(s, a, r, s_next, a_next, Q):
    eta = 0.1 # 학습 계수
    gamma = 0.9 # 시간 할인율

    if s_next == 8:
        Q[s,a] = Q[s,a] + eta * (r - Q[s,a])
    else:
        Q[s,a] = Q[s,a] + eta * (r + gamma * Q[s_next, a_next] - Q[s,a])
    return Q

# Q 학습에 따른 행동 가치 함수 갱신

def q_learning(s,a,r,s_next,a_next,Q):
    eta = 0.1 # 학습 계수
    gamma = 0.9 # 시간 할인율

    if s_next == 8:
        Q[s,a] = Q[s,a] + eta * (r - Q[s,a])
    else:
        Q[s,a] = Q[s,a] + eta * (r + gamma * np.nanmax(Q[s_next,:]) - Q[s,a])
    return Q


#%%

# 4-3-14 1 에피소드 실행

# 1 에피소드 실행

def play(Q, epsilon, pi):
    s = 0 # 상태
    
    a = a_next = get_a(s,Q,epsilon,pi) # 행동 초기값
    s_a_history = [[0,np.nan]] # 상태와 행동 이력

    # 에피소드 완료 시까지 반복
    while True:
        # 행동에 따른 다음 상태 얻기
        a = a_next
        s_next = get_s_next(s,a)

        # 이력 갱신
        s_a_history[-1][1] = a
        s_a_history.append([s_next, np.nan])

        # 종료 판정

        if s_next == 8:
            r = 1
            a_next = np.nan
        else:
            r = 0
            # 행동 가치 함수 Q 에 따라 행동 얻기
            a_next = get_a(s_next, Q, epsilon, pi)
            # print("s_next:{}\nQ: {}\nepsilon: {}\npi: {}\na_next: {}".format(s_next,Q[s_next,:],epsilon,pi[s_next,:],a_next))

        # 행동 가치 함수 갱신(Q 학습 시에는 Q learning으로 전환)
        Q = sarsa(s,a,r,s_next, a_next, Q)
        # Q = q_learning(s,a,r,s_next,a_next,Q)

        # 종료 판정
        if s_next == 8:
            break
        else:
            s = s_next
    
    # 이력과 행동 가치 함수 반환
    return [s_a_history, Q]

#%%

# 4-3-15 에피소드 반복 실행을 통한 학습

# epsilon greedy값을 점점 줄이기

epsilon = 0.5

# 에피소드를 반복해서 실행하며 학습
for episode in range(10):
    # epsilon greedy 값을 점점 감소시킴
    epsilon = epsilon / 2
    
    # 1 에피소드를 실행해 이력과 행동 가치 함수 추가
    [s_a_history, Q] = play(Q, epsilon, pi_0)

    # 출력
    print("에피소드: {}, 스텝: {}".format(episode, len(s_a_history) - 1))
    
#%%

def animate(i):
    state = s_a_history[i][0]
    circle.set_data((state%3) + 0.5, 2.5 - int(state / 3))
    return circle

anim = animation.FuncAnimation(fig, animate, frames=len(s_a_history), interval=200, repeat=False)
HTML(anim.to_jshtml())

