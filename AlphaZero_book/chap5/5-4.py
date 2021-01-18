# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:07:03 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%

# 이전 코드들


# 틱택토 생성
import random

# 게임 상태
class State:
    # 초기화
    def __init__(self, pieces=None, enemy_pieces=None):
        # 돌 배치
        self.pieces = pieces if pieces != None else [0]*9
        self.enemy_pieces = enemy_pieces if enemy_pieces != None else [0]*9

    # 돌의 수 취득
    def piece_count(self, pieces):
        count = 0
        for i in pieces:
            if i == 1:
                count += 1
        return count
    
    # 패배 여부 판정
    def is_lose(self):
        # 돌 3개 연결 여부
        def is_comp(x,y,dx,dy):
            for k in range(3):
                if y<0 or 2<y or x<0 or 2<x or \
                    self.enemy_pieces[x + y*3] == 0:
                    return False
                
                x = x + dx
                y = y + dy
            return True
        
        if is_comp(0,0,1,1) or is_comp(0,2,1,-1):
            return True
        
        for i in range(3):
            if is_comp(0,i,1,0) or is_comp(i,0,0,1):
                return True

        return False

    # 무승부 여부 판정
    def is_draw(self):
        return self.piece_count(self.pieces) + self.piece_count(self.enemy_pieces) == 9

    # 게임 종료 여부 판정
    def is_done(self):
        return self.is_lose() or self.is_draw()
    
    # 다음 상태 얻기
    def next(self,action):
        pieces = self.pieces.copy()
        pieces[action] = 1
        return State(self.enemy_pieces, pieces)

    # 둘 수 있는 수의 리스트 얻기
    def legal_actions(self):
        actions = []
        for i in range(9):
            if self.pieces[i] == 0 and self.enemy_pieces[i] == 0:
                actions.append(i)
        return actions
    
    # 선 수 여부 판정
    def is_first_player(self):
        return self.piece_count(self.pieces) == self.piece_count(self.enemy_pieces)
    
    # 문자열 표시
    def __str__(self):
        ox = ('o','x') if self.is_first_player() else ('x','o')
        str = ''

        for i in range(9):
            if self.pieces[i] == 1:
                str += ox[0]
            elif self.enemy_pieces[i] == 1:
                str += ox[1]
            else:
                str += '-'
            
            if i%3 == 2:
                str += '\n'
        return str


# 알파베타법을 활용한 상태 가치 계산
def alpha_beta(state, alpha, beta):
    # 패배 시 상태 가치 -1
    if state.is_lose():
        return -1
    
    # 무승부 시 상태 가치 0
    if state.is_draw():
        return 0
    
    # 둘 수 있는 수의 상태 가치 계산
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -beta, -alpha)
        if score > alpha:
            alpha = score
        
        # 현재 노드의 베스트 스코어가 새로운 노드보다 크면 탐색 종료
        if alpha >= beta:
            return alpha
        
    # 둘 수 있는 수의 상태 가치 최댓값 반환
    return alpha


# 5-2-5 알파베타법을 활용한 행동 선택

# 알파베타법을 활용한 행동 선택
def alpha_beta_action(state):
    # 둘 수 있는 수의 상태 가치 계산
    best_action = 0
    alpha = -float('inf')
    str = ['','']
    for action in state.legal_actions():
        score = -alpha_beta(state.next(action), -float('inf'), -alpha)
        if score > alpha:
            best_action = action
            alpha = score
        
        str[0] = "{}{:2d},".format(str[0], action)
        str[1] = "{}{:2d},".format(str[1], score)
    #print('action: ',str[0],"\nscore: ",str[1],"\n")

    # 둘 수 있는 수의 상태 가치값 중 최댓값을 선택하는 행동 반환
    return best_action

def random_action(state):
    legal_actions = state.legal_actions()
    return legal_actions[random.randint(0,len(legal_actions)-1)]

def playout(state):
    # 패배 시 상태 가치 -1
    if state.is_lose():
        return -1
    
    # 무승부 시 상태 가치 0
    if state.is_draw():
        return 0
    
    # 다음 상태의 상태 평가
    return -playout(state.next(random_action(state)))

#%%

import math
import numpy as np

# 몬테카를로 트리 서치의 노드 정의
class Node:
    def __init__(self, state):
        self.state = state # 상태
        self.w = 0 # 보상 누계
        self.n = 0 # 시행 횟수
        self.child_nodes = None # 자식 노드

    # 국면 가치 계산
    def evaluate(self):
        # 게임 종료 시
        if self.state.is_done():
            # 승패 결과로 가치 취득
            value = -1 if self.state.is_lose() else 0 # 패배 시 -1 무승부 시 0

            # 보상 누계와 시행 횟수 갱신
            self.w += value
            self.n += 1
            return value
            
        # 자식 노드가 존재하지 않는 경우
        if not self.child_nodes:
            # 플레이아웃으로 가치 얻기
            value = playout(self.state)

            # 보상 누계와 시행 횟수 갱신
            self.w += value
            self.n += 1

            # 자식 노드 전개
            if self.n == 10:
                self.expand()
            return value

        # 자식 노드가 존재하는 경우
        else:
            # UCB1이 가장 큰 자식 노드를 평가해 가치 얻기
            value = -self.next_child_node().evaluate()

            # 보상 누계와 시행 횟수 갱신
            self.w += value
            self.n += 1
            return value
    
    # 자식 노드 전개
    def expand(self):
        legal_actions = self.state.legal_actions()
        self.child_nodes = []
        for action in legal_actions:
            self.child_nodes.append(Node(self.state.next(action)))
    
    # UCB1이 가장 큰 자식 노드 얻기
    def next_child_node(self):
        # 시행 횟수가 0인 자식 노드 반환
        for child_node in self.child_nodes:
            if child_node.n == 0:
                return child_node
        
        # UCB1 계산
        t = 0
        for c in self.child_nodes:
            t += c.n
        
        ucb1_values = []
        for child_node in self.child_nodes:
            ucb1_values.append(-child_node.w / child_node.n + (2 * math.log(t)/child_node.n)**0.5)
        
        return self.child_nodes[np.argmax(ucb1_values)]


# 몬테카를로 트리 서치의 행동 선택
def mcts_action(state):
    
                
    root_node = Node(state)
    root_node.expand()

    # 100회 시뮬레이션 진행
    for _ in range(1000):
        root_node.evaluate()
    
    # 시행 횟수가 가장 큰 값을 갖는 행동 반환
    legal_actions = state.legal_actions()
    n_list = []
    for c in root_node.child_nodes:
        n_list.append(c.n)
    return legal_actions[np.argmax(n_list)]

#%%

# 몬테카를로 트리 탐색 vs 랜덤/알파베타법 대전

# 파라미터
EP_GAME_COUNT = 100 # 평가 1회당 게임 수

# 선 수 플레이어 포인트
def first_player_point(ended_state):
    # 1: 선 수 플레이어 승리, 0: 선 수 플레이어 패배, 0.5 무승부
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    
    return 0.5

# 게임 실행
def play(next_actions):
    # 상태 생성
    state = State()

    # 게임 종료 시까지 반복
    while True:
        # 게임 종료 시
        if state.is_done():
            break
        
        # 행동 얻기
        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        # 다음 상태 얻기
        state = state.next(action)

        # 선 수 플레이어 포인트 반환
    return first_player_point(state)


# 임의의 알고리즘 평가
def evaluate_algorithm_of(label, next_actions):
    # 여러 차례 대전 반복
    total_point = 0
    for i in range(EP_GAME_COUNT):
        # 게임 실행
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))
        
        # 출력
        print("\rEvaluate {}/{}".format(i + 1, EP_GAME_COUNT),end='')    
    print("")

    # 평균 포인트 계산
    average_point = total_point / EP_GAME_COUNT
    print(label.format(average_point))

# vs 랜덤
next_actions = (mcts_action, random_action)
evaluate_algorithm_of('VS Random {:.3f}', next_actions)

# vs 알파베타법
next_actions = (mcts_action, alpha_beta_action)
evaluate_algorithm_of('VS AlphaBeta {:.3f}', next_actions)