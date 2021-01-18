# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 17:51:24 2021

@author: 알파제로를 분석하며 배우는 인공지능
"""

#%% 
# 2-3-1 문자열

print("Hello World!")

# 2-3-2 변수와 연산자
a=1
b=2
c = a + b
print(a,b,c)

#%%
# 2-3-2 연산자
a = 5
s = 'a가 10 이상' if a>= 10 else 'a는 10 미만'
print(s)

#%%
# 2-3-3 문자열
# 여러 행의 문자열
text = '''텍스트1번째,
텍스트2번째'''
print(text)

# 문자열 추출
text = "Hello World"
print(text[1:3])
print(text[:5])
print(text[6:])

# 문자열에 변수 대입
a = 'test'
b = 100
c = 3.14159

print('문자열 = {}'.format(a))
print('부동소수점수(소수점둘째자리까지 = {:.2f}'.format(c))
print("여러변수 = {},{},{}".format(a,b,c))

#%%

# 2-3-4 리스트
my_list = [1,2,3,4]
print(my_list)
print(my_list[0])
print(my_list[1:3])

# 리스트 엘리먼트 변경
my_list[1:4] = [20,30]
print(my_list)

# range()를 활용한 리스트 생성
print(list(range(10)))
print(list(range(1,7)))
print(list(range(1,10,2)))

#%%

# 2-3-5 딕셔너리

# 딕셔너리 생성과 엘리먼트 취득

my_dic = {'apple':300, 'cherry':200, 'strawberry':3000}

print(my_dic)

# 딕셔너리 엘리먼트 추가 및 삭제

my_dic['apple'] = 400
print(my_dic)

#%%

# 2-3-6 튜플
my_tuple = (1,2,3,4)
print(my_tuple[0])

#%%

# 2-3-7 제어 구문

# 이프문
print('이프문')
num = 5

if num >= 10:
    print("10이상")
else:
    print("10이하")

# 여러 이프문
print('여러이프문')
num = 10
if num <5 :
    print("5")
elif num >=5:
    print("55")
else:
    print("none")

# for 반복문
print("for 반복문")
for n in [1,2,3]:
    print(n)
    print(n*10)

for n in range(5):
    print(n)


# while 반복문
print("while 반복문")
i = 0
while i<20:
    i+=1
    if i % 2 == 0:
        continue
    
    if i % 3 == 0:
        print(i)
    
# enumerate 열거형
print("enumerate 열거형")
for num, fruit in enumerate(['a','b','c']):
    print("{}:{}".format(num,fruit))

# 리스트 컴프리헨션
print("리스트 컴프리헨션")
my_list = []
for x in range(10):
    my_list.append(x*2)

print(my_list)

my_list2 = [x*2for x in range(10)]
print(my_list2)

#%%

# 2-3-8 함수와 람다식

# 함수
print("함수")

def radian(x):
    return x / 180 * 3.14159

for x in range(0,360,90):
    print("각도 : {} / 라디안 : {}".format(x,radian(x)))


# 람다식
print("람다식")

lambda_radian = (lambda x:x/180*3.14159)

for x in range(0,360,90):
    print("각도 : {} / 라디안 : {}".format(x,lambda_radian(x)))
    
#%%

# 클래스
print("클래스")

class HelloClass:
    def __init__(self,msg):
        self.msg = msg
    
    def output(self):
        print(self.msg)

hello = HelloClass('Hello World')
hello.output()

#%%

# 2-3-10 패키지 임포트와 컴포넌트 직접 호출

#패키지 임포트
print("패키지 임포트")
import numpy as np

a = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)

#컴포넌트 직접 호출
print("컴포넌트 직접 호출")
from numpy import array

a = array([[1,2,3],[4,5,6],[7,8,9]])
print(a)

