---
layout: single
title: "2024 생명연구자원 AI활용 경진대회 : 인공지능 활용 부문 대회 참가 기록"
excerpt: "생명연구자원 AI활용 경진대회 코드 및 성찰"
mathjax: true
toc: true
toc_sticky: true
toc_label: Code
categories: Dacon
tag: [AI, ML, Competition]
---

## Domain Knowledge
먼저 

## Data Structure 분석
대회 준비를 기간 초반부터 하지 못했었는데 토크 페이지를 보니 데이터 불균형과 훈련용 데이터에 없는 데이터(?)가 test 데이터에 있다는 말들이 있었다. 그래서 원래도 해야되지만 데이터의 구조를 먼저 파악하기 위해 class와 데이터들을 확인했다.

### 라이브러리 호출

```python
import pandas as pd
import numpy as np
```

### Train data 확인
훈련용 데이터를 엑셀로 열어보니 한 칸에 여러개의 값이 들어가 있는 것을 발견했다. list에 WT가 아닌 모든 값을 넣은 후 중복을 제거해 개수를 파악했다.

```python
train = pd.read_csv("train.csv")
train2 = train.drop(columns=['ID', 'SUBCLASS']).astype(str)

adj_train = []
for i in range(train2.shape[0]):
    for ii in range(train2.shape[1]):
        tmp = np.char.split(train2.values[i,ii]).item()[0]
        if tmp != 'WT':
            adj_train.append(tmp)
adj_train = list(set(adj_train))
print('train.csv 변이 정보 개수:', len(adj_train))
```

train.csv 변이 정보 개수: 104605

### Test data 확인
테스트 데이터 또한 같은 작업을 해주었다.

```python
test = pd.read_csv("test.csv")
test2 = test.drop(columns=['ID']).astype(str)

adj_test = []
for i in range(test2.shape[0]):
    for ii in range(test2.shape[1]):
        tmp = np.char.split(test2.values[i,ii]).item()[0]
        if tmp != 'WT':
            adj_test.append(tmp)
adj_test = list(set(adj_test))
print('test.csv 변이 정보 개수:', len(adj_test))
```

test.csv 변이 정보 개수: 92093

### Test data 에만 있는 중류 수 확인
각각에 서로 있는 것과 없는 것이 다를 수 있으므로 무작정 train data 변이정보 개수 - test data 변이정보 개수를 할 수 없다. 문제가 되는 상황은 훈련 시 한 번도 보지 못한 정보를 테스트 시 보게 되는 경우 이므로 이런 상황에 해당하는 정보 개수를 파악했다.

```python
only_test=[]
for k in adj_test:
    if k not in adj_train:
        only_test.append(k)

print('test.csv 에만 있는 변이 정보 개수:',len(only_test))
```
