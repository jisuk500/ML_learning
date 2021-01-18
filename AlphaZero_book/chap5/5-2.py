# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 18:06:56 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%%

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
    
# 5-1-5 미니맥스법을 활용한 상태 가치 계산 함수

# 미니맥스법을 활용한 상태 가치 계산
def mini_max(state):
    # 패배 시 상태 가치 -1
    if state.is_lose():
        return -1
    
    # 무승부 시 상태 가치 0
    if state.is_draw():
        return 0
    
    # 둘 수 있는 수의 상태 가치 계산
    best_score = -float('inf')
    for action in state.legal_actions():
        score = -mini_max(state.next(action))
        if score > best_score:
            best_score = score
    
    # 둘 수 있는 수의 상태 가치값 중 최댓값 선택
    return best_score
    
# 미니 맥스 버블 활용한 행동 선택
def mini_max_action(state):
    # 둘 수 있는 수의 상태 가치 계산
    best_action = 0
    best_score = -float('inf')
    str = ["",""]
    for action in state.legal_actions():
        score = -mini_max(state.next(action))
        if score > best_score:
            best_action = action
            best_score = score

        str[0] = "{}{:2d},".format(str[0], action)
        str[1] = "{}{:2d},".format(str[1], score)
    print("action:", str[0],"\nscore:",str[1],'\n')

    # 둘 수 있는 수의 상태 가치 최댓값을 가진 행동 반환
    return best_action

#%%

# 조금 개선한 미니맥스법을 활용한 상태 가치 계산
def mini_max_plus(state, limit):
    # 패배 시 상태 가치 -1
    if state.is_lose():
        return -1
    
    # 무승부 시 상태 가치 0
    if state.is_draw():
        return 0
    
    # 둘 수 있는 수의 상태 가치 계산
    best_score = -float('inf')
    for action in state.legal_actions():
        score = -mini_max_plus(state.next(action), -best_score)
        if score > best_score:
            best_score = score
        
        # 현재 노드의 베스트 스코어가 새로운 노드보다 크면 탐색 종료
        if best_score >= limit:
            return best_score
    
    # 둘 수 있는 수의 상태 가치 최댓값을 반환
    return best_score

#%%

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

#%%

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
    print('action: ',str[0],"\nscore: ",str[1],"\n")

    # 둘 수 있는 수의 상태 가치값 중 최댓값을 선택하는 행동 반환
    return best_action

#%%

# 알파베타법과 미니맥스법의 대전

# 상태 생성
state = State()

# 게임 종료 시까지 반복
while True:
    # 게임 종료 시
    if state.is_done():
        break
    
    # 행동 얻기
    if state.is_first_player():
        action = alpha_beta_action(state)
    else:
        action = mini_max_action(state)
    
    # 다음 상태 얻기
    state = state.next(action)

    # 문자열 표시
    print(state)
    print()
