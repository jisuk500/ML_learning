# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:05:03 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%

# 4-4-4 패키지 임포트
import gym 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

from tensorflow.compat.v1.losses import huber_loss

# 4-4-5 파라미터 준비

# 파라미터 준비
NUM_EPISODES = 500 # 에피소드 수
MAX_STEPS = 200 # 최대 스텝 수
GAMMA = 0.99 # 시간 할인율
WARMUP = 10 # 초기화 시 조작하지 않는 스텝 수

# 검색 파라미터
E_START = 1.0 # 입실론 초기화
E_STOP = 0.01 # 입실론 최종값
E_DECAY_RATE = 0.001 # 입실론 감쇄율

# 메모리 파라미터
MEMORY_SIZE = 10000 # 경험 메모리 사이즈
BATCH_SIZE = 32 # 배치 사이즈

#%%

# 행동 평가 함수 정의
class QNetwork:
    # 초기화
    def __init__(self, state_size, action_size):
        # 모델 생성
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=state_size))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        # 모델 컴파일
        self.model.compile(loss=huber_loss, optimizer=Adam(lr=0.001))


#%%

# 4-4-7 경험 메모리 정의

# 경험 메모리 클래스 정의
class Memory():
    # 초기화
    def __init__(self, memory_size):
        self.buffer = deque(maxlen=memory_size)
    
    # 경험 추가
    def add(self, experience):
        self.buffer.append(experience)
    
    # 배치 사이즈만큼의 경험을 랜덤으로 얻음
    def sample(self,batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]
    
    # 경험 메모리 사이즈
    def __len__(self):
        return len(self.buffer)
    
#%%

# 4-4-8 환경 생성

# 환경 생성
env = gym.make('CartPole-v0')
state_size = env.observation_space.shape[0] # 행동 수
action_size = env.action_space.n # 상태 수

# 4-4-9 메인 네트워크, 대상 네트워크 및 경험 메모리 생성
# 메인 네트워크 생성
main_qn = QNetwork(state_size, action_size)

# 대상 네트워크 생성
target_qn = QNetwork(state_size, action_size)

# 경험 메모리 생성
memory = Memory(MEMORY_SIZE)

#%%

# 환경 초기화
state = env.reset()
state = np.reshape(state, [1, state_size])

# 에피소드 수만큼 에피소드 반복
total_step = 0 # 총 스텝 수
success_count = 0 # 성공 수

for episode in range(1, NUM_EPISODES):
    step = 0 # 스텝 수

    # 대상 네트워크 갱신
    target_qn.model.set_weights(main_qn.model.get_weights())

    # 1 에피소드 루프
    for iii in range(1, MAX_STEPS + 1):
        step += 1
        total_step += 1

        # 입실론을 감소시킴
        epsilon = E_STOP + (E_START - E_STOP) * np.exp(-E_DECAY_RATE * total_step/2)

        # 랜덤하게 행동 선택
        if epsilon > np.random.rand():
            action = env.action_space.sample()
        else:
            action = np.argmax(main_qn.model.predict(state)[0])
        
        # 행동에 맞추어 상태와 보상을 얻음
        next_state, _, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        print('{}번째 스텝'.format(iii))

        # 에피소드 완료 시
        if done:
            # 보상 지정
            if step>= 190:
                success_count += 1
                reward = 1
            else:
                success_count = 0
                reward = 0
            
            # 다음 상태에 상태 없음을 대입
            next_state = np.zeros(state.shape)

            if step > WARMUP:
                memory.add((state, action, reward, next_state))
            
        # 에피소드 미완료 시
        else:
            # 보상 지정
            reward = 0

            # 경험 추가
            if step > WARMUP:
                memory.add((state, action,reward, next_state))
            
            # 상태에 다음 상태 대입
            state = next_state

        # 에피소드 완료 시
        if done:
            # 에피소드 루프 이탈
            break
    
    # 에피소드 완료 시 로그 표시
    print("에피소드: {}, 스텝 수: {}, epsilon: {:.4f}".format(episode, step, epsilon))


     # 행동 평가 함수 갱신
    if len(memory) >= BATCH_SIZE:
        # 뉴럴 네트워크의 입력과 출력 준비
        inputs = np.zeros((BATCH_SIZE,4)) # 입력(상태)
        targets = np.zeros((BATCH_SIZE,2)) # 출력(행동별 가치)

        # 배치 사이즈 만큼 경험을 랜덤하게 선택
        minibatch = memory.sample(BATCH_SIZE)

        # 뉴럴 네트워크 입력과 출력 생성
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):

            # 입력 상태 지정
            inputs[i] = state_b

            # 선택한 행동의 가치 계산
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                target = reward_b + GAMMA * np.amax(target_qn.model.predict(next_state_b)[0])
            else:
                target = reward_b
                
            # 출력에 행동별 가치 지정
            targets[i] = main_qn.model.predict(state_b)
            targets[i][action_b] = target # 선택한 행동 가치
            # print('메모리 {}번째 계산'.format(i))
            
        # 행동 가치 함수 갱신
        main_qn.model.fit(inputs, targets, epochs=1, verbose=0)
    

    # 5 회 연속 성공으로 학습 완료
    if success_count >= 5:
        break
    
    # 환경 초기화
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    
#%%

# 디스플레이 설정 인스톨
# 4-4-16 애니메이션 프레임 생성

# 평가
frames = [] # 애니메이션 프레임

# 환경 초기화
state = env.reset()
state = np.reshape(state, [1, state_size])

# 1 에피소드 루프
step = 0 # 스텝 수
for _ in range(1, MAX_STEPS+1):
    step += 1

    # 애니메이션 프레임 추가
    frames.append(env.render(mode='rgb_array'))

    # 최적 행동 선택
    action = np.argmax(main_qn.model.predict(state)[0])

    # 행동에 맞추어 상황과 보상을 얻음
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1 ,state_size])

    # 에피소드 완료 시
    if done:   
        # 다음 상태에 상태 없음을 대입
        next_state = np.zeros(state.shape)

        # 에피소드 루프 이탈
        break
    else:
        # 상태에 다음 상태를 대입
        state = next_state
    
# 에피소드 완료 시 로그 표시
print('스텝 수: {}'.format(step))




