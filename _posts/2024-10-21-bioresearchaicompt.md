---
layout: single
title: "2024 생명연구자원 AI활용 경진대회 : 인공지능 활용 부문 대회 참가 기록"
excerpt: "생명연구자원 AI활용 경진대회 코드 및 성찰"
mathjax: true
toc: true
toc_sticky: true
toc_label: Code
categories: Dacon
tag: [AI, ML, Competition, Bio]
---

## Domain Knowledge
훈련용 데이터셋과 추론용 데이터셋에 등장하는 데이터들은 암환자 유전체 데이터의 변이 정보이다. 이 데이터에는 다양한 변이 정보들이 있다.

### 1. 미스센스 변이 (Missense Mutation)
정의: 하나의 염기서열 변화로 인해, 한 아미노산이 다른 아미노산으로 바뀌는 변이이다. \\
변이 예시로는 R132H 와 같이 맨 앞과 맨 뒤의 알파벳이 다른 구조로, 암세포에서 발생하는 대표적인 변이.

### 2. 동의 변이 (Synonymous Mutation)
정의: 염기서열이 변했지만, 변이된 코돈이 여전히 같은 아미노산을 암호화하여 단백질 서열에 변화가 없는 변이. \\
변이 예시로는 I752I 와 같이 맨 앞과 맨 뒤의 알파벳이 같은 구조이다.

### 3. 넌센스 변이 (Nonsense Mutation)
정의: 아미노산을 암호화하던 코돈이 변이로 인해 종결 코돈(Stop codon)이 되어, 번역이 조기에 중단되는 변이.\\
변이 예시로는 W22X 와 같이 W(트립토판)이 X(종결코돈)으로 변이하는 구조이다.

### 4. 결실 변이 (Deletion Mutation)
정의: 염기서열에서 하나 이상의 염기가 삭제되어 변이가 발생.\\
변이 예시로는 A123* 과 같이 맨 뒤의 알파벳이 *로 되어있다.

### 5. 복합 변이 (Complex Mutation)
정의: 두 가지 이상의 변이가 함께 일어나는 경우.\\
변이 예시로는 R132H V147A 와 같이 여러 가지의 변이가 함께 일어나는 경우이다.

### 6. 프레임 쉬프트 변이 (Frameshift Mutation)
정의: 염기서열에서 삽입(Insertion) 또는 결실(Deletion)로 인해, 코돈 읽기 틀이 변화하여 전체 서열이 바뀌는 변이. \\
변이 예시로는 L31fs 와 같이 from 은 있는데, to 가 없어 아미노산 서열이 완전히 달라지고, 비정상적인 단백질이 생성되는 경우이다. 

(출처: 대회 내 토크 페이지 - 단백질 변이(유전적 변이)에 대한 배경 지식 공유 https://dacon.io/competitions/official/236355/talkboard/412868?page=1&dtype=recent)

## 코드
먼저 필요한 라이브러리를 호출한다.
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import string
import itertools
```

### Data Structure 분석
대회 준비를 기간 초반부터 하지 못했었는데 토크 페이지를 보니 데이터 불균형과 훈련용 데이터에 없는 데이터(?)가 테스트 데이터에 있다는 말들이 있었다. 그래서 원래도 해야되지만 데이터의 구조를 먼저 파악하기 위해 class와 데이터들을 확인했다.
훈련용 데이터를 엑셀로 열어보니 한 칸에 여러 값이 들어가 있는 것을 발견했다. list에 WT가 아닌 모든 값을 넣은 후 중복을 제거해 개수를 파악했다.

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

그러면 결과는 아래와 같이 나온다. \\
test.csv 에만 있는 변이 정보 개수: 46622

### 아미노산 변이 정보 매핑
도메인 지식이 부족해서 교수님께 조언을 구했는데, 변이 정보를 먼저 매핑을 하는게 좋겠다고 하셨다. 변이 정보에는 (알파벳)(약 3자리 숫자)(알파벳) 형태를 갖고 있는데, from to에 포커스를 맞춰서 매핑하기로 했다. 변이가 일어나지 않은 WT와 from to 가 같은 A->A 와 같은 동의 변이의 경우에서는 0으로, 나머지 A->B와 같은 경우들은 순서대로 숫자를 부여했다.
먼저 조합표를 만들었다.

```python
alphabet1, alphabet2 = list(string.ascii_uppercase), list(string.ascii_uppercase)
combination = ['WT']
for a1 in alphabet1:
    for a2 in alphabet2:
        if a1+a2 != 'WT':
            combination.append(a1+a2)
print(combination)
```

#### 조합표 생성 결과
```python
['WT', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL', 'CM', 'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DJ', 'DK', 'DL', 'DM', 'DN', 'DO', 'DP', 'DQ', 'DR', 'DS', 'DT', 'DU', 'DV', 'DW', 'DX', 'DY', 'DZ', 'EA', 'EB', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EJ', 'EK', 'EL', 'EM', 'EN', 'EO', 'EP', 'EQ', 'ER', 'ES', 'ET', 'EU', 'EV', 'EW', 'EX', 'EY', 'EZ', 'FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI', 'FJ', 'FK', 'FL', 'FM', 'FN', 'FO', 'FP', 'FQ', 'FR', 'FS', 'FT', 'FU', 'FV', 'FW', 'FX', 'FY', 'FZ', 'GA', 'GB', 'GC', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GJ', 'GK', 'GL', 'GM', 'GN', 'GO', 'GP', 'GQ', 'GR', 'GS', 'GT', 'GU', 'GV', 'GW', 'GX', 'GY', 'GZ', 'HA', 'HB', 'HC', 'HD', 'HE', 'HF', 'HG', 'HH', 'HI', 'HJ', 'HK', 'HL', 'HM', 'HN', 'HO', 'HP', 'HQ', 'HR', 'HS', 'HT', 'HU', 'HV', 'HW', 'HX', 'HY', 'HZ', 'IA', 'IB', 'IC', 'ID', 'IE', 'IF', 'IG', 'IH', 'II', 'IJ', 'IK', 'IL', 'IM', 'IN', 'IO', 'IP', 'IQ', 'IR', 'IS', 'IT', 'IU', 'IV', 'IW', 'IX', 'IY', 'IZ', 'JA', 'JB', 'JC', 'JD', 'JE', 'JF', 'JG', 'JH', 'JI', 'JJ', 'JK', 'JL', 'JM', 'JN', 'JO', 'JP', 'JQ', 'JR', 'JS', 'JT', 'JU', 'JV', 'JW', 'JX', 'JY', 'JZ', 'KA', 'KB', 'KC', 'KD', 'KE', 'KF', 'KG', 'KH', 'KI', 'KJ', 'KK', 'KL', 'KM', 'KN', 'KO', 'KP', 'KQ', 'KR', 'KS', 'KT', 'KU', 'KV', 'KW', 'KX', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LD', 'LE', 'LF', 'LG', 'LH', 'LI', 'LJ', 'LK', 'LL', 'LM', 'LN', 'LO', 'LP', 'LQ', 'LR', 'LS', 'LT', 'LU', 'LV', 'LW', 'LX', 'LY', 'LZ', 'MA', 'MB', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MI', 'MJ', 'MK', 'ML', 'MM', 'MN', 'MO', 'MP', 'MQ', 'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 'MZ', 'NA', 'NB', 'NC', 'ND', 'NE', 'NF', 'NG', 'NH', 'NI', 'NJ', 'NK', 'NL', 'NM', 'NN', 'NO', 'NP', 'NQ', 'NR', 'NS', 'NT', 'NU', 'NV', 'NW', 'NX', 'NY', 'NZ', 'OA', 'OB', 'OC', 'OD', 'OE', 'OF', 'OG', 'OH', 'OI', 'OJ', 'OK', 'OL', 'OM', 'ON', 'OO', 'OP', 'OQ', 'OR', 'OS', 'OT', 'OU', 'OV', 'OW', 'OX', 'OY', 'OZ', 'PA', 'PB', 'PC', 'PD', 'PE', 'PF', 'PG', 'PH', 'PI', 'PJ', 'PK', 'PL', 'PM', 'PN', 'PO', 'PP', 'PQ', 'PR', 'PS', 'PT', 'PU', 'PV', 'PW', 'PX', 'PY', 'PZ', 'QA', 'QB', 'QC', 'QD', 'QE', 'QF', 'QG', 'QH', 'QI', 'QJ', 'QK', 'QL', 'QM', 'QN', 'QO', 'QP', 'QQ', 'QR', 'QS', 'QT', 'QU', 'QV', 'QW', 'QX', 'QY', 'QZ', 'RA', 'RB', 'RC', 'RD', 'RE', 'RF', 'RG', 'RH', 'RI', 'RJ', 'RK', 'RL', 'RM', 'RN', 'RO', 'RP', 'RQ', 'RR', 'RS', 'RT', 'RU', 'RV', 'RW', 'RX', 'RY', 'RZ', 'SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SJ', 'SK', 'SL', 'SM', 'SN', 'SO', 'SP', 'SQ', 'SR', 'SS', 'ST', 'SU', 'SV', 'SW', 'SX', 'SY', 'SZ', 'TA', 'TB', 'TC', 'TD', 'TE', 'TF', 'TG', 'TH', 'TI', 'TJ', 'TK', 'TL', 'TM', 'TN', 'TO', 'TP', 'TQ', 'TR', 'TS', 'TT', 'TU', 'TV', 'TW', 'TX', 'TY', 'TZ', 'UA', 'UB', 'UC', 'UD', 'UE', 'UF', 'UG', 'UH', 'UI', 'UJ', 'UK', 'UL', 'UM', 'UN', 'UO', 'UP', 'UQ', 'UR', 'US', 'UT', 'UU', 'UV', 'UW', 'UX', 'UY', 'UZ', 'VA', 'VB', 'VC', 'VD', 'VE', 'VF', 'VG', 'VH', 'VI', 'VJ', 'VK', 'VL', 'VM', 'VN', 'VO', 'VP', 'VQ', 'VR', 'VS', 'VT', 'VU', 'VV', 'VW', 'VX', 'VY', 'VZ', 'WA', 'WB', 'WC', 'WD', 'WE', 'WF', 'WG', 'WH', 'WI', 'WJ', 'WK', 'WL', 'WM', 'WN', 'WO', 'WP', 'WQ', 'WR', 'WS', 'WU', 'WV', 'WW', 'WX', 'WY', 'WZ', 'XA', 'XB', 'XC', 'XD', 'XE', 'XF', 'XG', 'XH', 'XI', 'XJ', 'XK', 'XL', 'XM', 'XN', 'XO', 'XP', 'XQ', 'XR', 'XS', 'XT', 'XU', 'XV', 'XW', 'XX', 'XY', 'XZ', 'YA', 'YB', 'YC', 'YD', 'YE', 'YF', 'YG', 'YH', 'YI', 'YJ', 'YK', 'YL', 'YM', 'YN', 'YO', 'YP', 'YQ', 'YR', 'YS', 'YT', 'YU', 'YV', 'YW', 'YX', 'YY', 'YZ', 'ZA', 'ZB', 'ZC', 'ZD', 'ZE', 'ZF', 'ZG', 'ZH', 'ZI', 'ZJ', 'ZK', 'ZL', 'ZM', 'ZN', 'ZO', 'ZP', 'ZQ', 'ZR', 'ZS', 'ZT', 'ZU', 'ZV', 'ZW', 'ZX', 'ZY', 'ZZ']
```

### Data Preprocessing
위에서 확인했듯이 하나의 속성에 여러 값이 들어가 있는 경우, 즉 복합 변이가 있으므로 이를 먼저 해결해야한다. 제 1정규화를 해야하는데 여기서 고민이 많이 되었다. 속성 하나에 값이 하나인 것들은 복사하고 여러개인 것들은 나눠서 오버샘플링처럼 해야 되나, 여러개의 속성값들 중 하나만 남기는 언더샘플링을 해야 하나 말이다.
내가 내린 결론은 일단 언더샘플링이었다. 복합 변이라서 함부로 값을 나누거나 하나만 남기고 날리면 정보의 손실이 크겠지만 XGBoost를 사용해서 학습을 시키기 위해서는 object 형식은 불가능하기에 값을 하나만 남겨야만 했다. 그래서 추후에 성능이 안나오면 object 형태의 데이터도 학습이 가능한 모델로 변경하거나 전처리단에서부터 자연어 처리의 방향으로 진행하는 계획도 세웠다. 
먼저 전처리 전에 subclass 라벨 인코딩을 한 이후, 전처리를 진행한다. 전처리 과정에서는 위에서 만든 조합표에 훈련 데이터를 맵핑한다. 속성 하나에 값이 여러개인 경우는 첫번째 값을 사용하도록 했다.

```python
train = pd.read_csv("train.csv", header=0)

le_subclass = LabelEncoder()
train['SUBCLASS'] = le_subclass.fit_transform(train['SUBCLASS'])
for i, label in enumerate(le_subclass.classes_):
    print(f"원래 레이블: {label}, 변환된 숫자: {i}")

X = train.drop(columns=['SUBCLASS', 'ID']).copy()
y_subclass = train['SUBCLASS']

def preprocessing_train(element):
    # 띄어쓰기로 분리되어있는 element는 리스트로 쪼개기
    if isinstance(element, str) and ' ' in element:
        element = element.split()
    
    # 값이 한 개(문자열)인 경우
    if isinstance(element, str):
        if element[0] + element[-1] not in combination:
            combination.append(element[0]+element[-1])  ## 알파벳이 아닌 * 와 같은 문자도 있는 것을 발견하고 조합표에 없는 경우 추가하는 코드를 넣었다.
        return np.where(element[0] == element[-1], 0, combination.index(element[0] + element[-1]))
        
    # 값이 여러개(리스트)인 경우
    elif isinstance(element, list):
         return preprocessing_train(element[0])

X_prep = X.applymap(preprocessing_train)

print(X_prep)
print('업데이트된 조합표')
print(combination)
X_prep.to_csv('preprocessing_data.csv')
```

#### 전처리 결과는 아래와 같다.
```python
원래 레이블: ACC, 변환된 숫자: 0
원래 레이블: BLCA, 변환된 숫자: 1
원래 레이블: BRCA, 변환된 숫자: 2
원래 레이블: CESC, 변환된 숫자: 3
원래 레이블: COAD, 변환된 숫자: 4
원래 레이블: DLBC, 변환된 숫자: 5
원래 레이블: GBMLGG, 변환된 숫자: 6
원래 레이블: HNSC, 변환된 숫자: 7
원래 레이블: KIPAN, 변환된 숫자: 8
원래 레이블: KIRC, 변환된 숫자: 9
원래 레이블: LAML, 변환된 숫자: 10
원래 레이블: LGG, 변환된 숫자: 11
원래 레이블: LIHC, 변환된 숫자: 12
원래 레이블: LUAD, 변환된 숫자: 13
원래 레이블: LUSC, 변환된 숫자: 14
원래 레이블: OV, 변환된 숫자: 15
원래 레이블: PAAD, 변환된 숫자: 16
원래 레이블: PCPG, 변환된 숫자: 17
원래 레이블: PRAD, 변환된 숫자: 18
원래 레이블: SARC, 변환된 숫자: 19
원래 레이블: SKCM, 변환된 숫자: 20
원래 레이블: STES, 변환된 숫자: 21
원래 레이블: TGCT, 변환된 숫자: 22
원래 레이블: THCA, 변환된 숫자: 23
원래 레이블: THYM, 변환된 숫자: 24
원래 레이블: UCEC, 변환된 숫자: 25

      A2M  AAAS  AADAT  AARS1  ABAT  ABCA1  ABCA2  ABCA3  ABCA4  ABCA5  ...  \
0       0     0      0      0     0      0      0      0      0      0  ...   
1       0     0      0      0     0      0      0      0      0      0  ...   
2       0     0      0      0     0      0      0      0      0      0  ...   
3       0     0      0      0     0      0      0      0      0      0  ...   
4       0     0      0      0     0      0      0      0      0      0  ...   
...   ...   ...    ...    ...   ...    ...    ...    ...    ...    ...  ...   
6196    0     0      0      0     0      0      0      0      0      0  ...   
6197    0     0      0      0     0      0      0      0      0      0  ...   
6198    0     0      0      0     0      0      0      0      0      0  ...   
6199    0     0      0      0     0      0      0      0      0      0  ...   
6200    0     0      0      0     0      0      0      0      0      0  ...   

      ZNF292  ZNF365  ZNF639  ZNF707  ZNFX1  ZNRF4  ZPBP  ZW10  ZWINT  ZYX  
0          0       0       0       0      0      0     0     0      0    0  
1          0       0       0       0      0      0     0     0      0    0  
2          0       0       0       0      0      0     0     0      0    0  
3          0       0       0       0      0      0     0     0      0    0  
4          0       0       0       0      0      0     0     0      0    0  
...      ...     ...     ...     ...    ...    ...   ...   ...    ...  ...  
6196       0       0       0       0      0      0     0     0      0    0  
6197       0       0       0       0      0      0     0     0      0    0  
6198       0       0       0       0      0      0     0     0    513    0  
6199       0       0       0       0      0      0     0     0      0    0  
6200       0       0       0       0      0      0     0     0      0    0  

[6201 rows x 4384 columns]
업데이트된 조합표
['WT', 'AA', 'AB', 'AC', 'AD', 'AE', 'AF', 'AG', 'AH', 'AI', 'AJ', 'AK', 'AL', 'AM', 'AN', 'AO', 'AP', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW', 'AX', 'AY', 'AZ', 'BA', 'BB', 'BC', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BK', 'BL', 'BM', 'BN', 'BO', 'BP', 'BQ', 'BR', 'BS', 'BT', 'BU', 'BV', 'BW', 'BX', 'BY', 'BZ', 'CA', 'CB', 'CC', 'CD', 'CE', 'CF', 'CG', 'CH', 'CI', 'CJ', 'CK', 'CL', 'CM', 'CN', 'CO', 'CP', 'CQ', 'CR', 'CS', 'CT', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DA', 'DB', 'DC', 'DD', 'DE', 'DF', 'DG', 'DH', 'DI', 'DJ', 'DK', 'DL', 'DM', 'DN', 'DO', 'DP', 'DQ', 'DR', 'DS', 'DT', 'DU', 'DV', 'DW', 'DX', 'DY', 'DZ', 'EA', 'EB', 'EC', 'ED', 'EE', 'EF', 'EG', 'EH', 'EI', 'EJ', 'EK', 'EL', 'EM', 'EN', 'EO', 'EP', 'EQ', 'ER', 'ES', 'ET', 'EU', 'EV', 'EW', 'EX', 'EY', 'EZ', 'FA', 'FB', 'FC', 'FD', 'FE', 'FF', 'FG', 'FH', 'FI', 'FJ', 'FK', 'FL', 'FM', 'FN', 'FO', 'FP', 'FQ', 'FR', 'FS', 'FT', 'FU', 'FV', 'FW', 'FX', 'FY', 'FZ', 'GA', 'GB', 'GC', 'GD', 'GE', 'GF', 'GG', 'GH', 'GI', 'GJ', 'GK', 'GL', 'GM', 'GN', 'GO', 'GP', 'GQ', 'GR', 'GS', 'GT', 'GU', 'GV', 'GW', 'GX', 'GY', 'GZ', 'HA', 'HB', 'HC', 'HD', 'HE', 'HF', 'HG', 'HH', 'HI', 'HJ', 'HK', 'HL', 'HM', 'HN', 'HO', 'HP', 'HQ', 'HR', 'HS', 'HT', 'HU', 'HV', 'HW', 'HX', 'HY', 'HZ', 'IA', 'IB', 'IC', 'ID', 'IE', 'IF', 'IG', 'IH', 'II', 'IJ', 'IK', 'IL', 'IM', 'IN', 'IO', 'IP', 'IQ', 'IR', 'IS', 'IT', 'IU', 'IV', 'IW', 'IX', 'IY', 'IZ', 'JA', 'JB', 'JC', 'JD', 'JE', 'JF', 'JG', 'JH', 'JI', 'JJ', 'JK', 'JL', 'JM', 'JN', 'JO', 'JP', 'JQ', 'JR', 'JS', 'JT', 'JU', 'JV', 'JW', 'JX', 'JY', 'JZ', 'KA', 'KB', 'KC', 'KD', 'KE', 'KF', 'KG', 'KH', 'KI', 'KJ', 'KK', 'KL', 'KM', 'KN', 'KO', 'KP', 'KQ', 'KR', 'KS', 'KT', 'KU', 'KV', 'KW', 'KX', 'KY', 'KZ', 'LA', 'LB', 'LC', 'LD', 'LE', 'LF', 'LG', 'LH', 'LI', 'LJ', 'LK', 'LL', 'LM', 'LN', 'LO', 'LP', 'LQ', 'LR', 'LS', 'LT', 'LU', 'LV', 'LW', 'LX', 'LY', 'LZ', 'MA', 'MB', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MI', 'MJ', 'MK', 'ML', 'MM', 'MN', 'MO', 'MP', 'MQ', 'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 'MZ', 'NA', 'NB', 'NC', 'ND', 'NE', 'NF', 'NG', 'NH', 'NI', 'NJ', 'NK', 'NL', 'NM', 'NN', 'NO', 'NP', 'NQ', 'NR', 'NS', 'NT', 'NU', 'NV', 'NW', 'NX', 'NY', 'NZ', 'OA', 'OB', 'OC', 'OD', 'OE', 'OF', 'OG', 'OH', 'OI', 'OJ', 'OK', 'OL', 'OM', 'ON', 'OO', 'OP', 'OQ', 'OR', 'OS', 'OT', 'OU', 'OV', 'OW', 'OX', 'OY', 'OZ', 'PA', 'PB', 'PC', 'PD', 'PE', 'PF', 'PG', 'PH', 'PI', 'PJ', 'PK', 'PL', 'PM', 'PN', 'PO', 'PP', 'PQ', 'PR', 'PS', 'PT', 'PU', 'PV', 'PW', 'PX', 'PY', 'PZ', 'QA', 'QB', 'QC', 'QD', 'QE', 'QF', 'QG', 'QH', 'QI', 'QJ', 'QK', 'QL', 'QM', 'QN', 'QO', 'QP', 'QQ', 'QR', 'QS', 'QT', 'QU', 'QV', 'QW', 'QX', 'QY', 'QZ', 'RA', 'RB', 'RC', 'RD', 'RE', 'RF', 'RG', 'RH', 'RI', 'RJ', 'RK', 'RL', 'RM', 'RN', 'RO', 'RP', 'RQ', 'RR', 'RS', 'RT', 'RU', 'RV', 'RW', 'RX', 'RY', 'RZ', 'SA', 'SB', 'SC', 'SD', 'SE', 'SF', 'SG', 'SH', 'SI', 'SJ', 'SK', 'SL', 'SM', 'SN', 'SO', 'SP', 'SQ', 'SR', 'SS', 'ST', 'SU', 'SV', 'SW', 'SX', 'SY', 'SZ', 'TA', 'TB', 'TC', 'TD', 'TE', 'TF', 'TG', 'TH', 'TI', 'TJ', 'TK', 'TL', 'TM', 'TN', 'TO', 'TP', 'TQ', 'TR', 'TS', 'TT', 'TU', 'TV', 'TW', 'TX', 'TY', 'TZ', 'UA', 'UB', 'UC', 'UD', 'UE', 'UF', 'UG', 'UH', 'UI', 'UJ', 'UK', 'UL', 'UM', 'UN', 'UO', 'UP', 'UQ', 'UR', 'US', 'UT', 'UU', 'UV', 'UW', 'UX', 'UY', 'UZ', 'VA', 'VB', 'VC', 'VD', 'VE', 'VF', 'VG', 'VH', 'VI', 'VJ', 'VK', 'VL', 'VM', 'VN', 'VO', 'VP', 'VQ', 'VR', 'VS', 'VT', 'VU', 'VV', 'VW', 'VX', 'VY', 'VZ', 'WA', 'WB', 'WC', 'WD', 'WE', 'WF', 'WG', 'WH', 'WI', 'WJ', 'WK', 'WL', 'WM', 'WN', 'WO', 'WP', 'WQ', 'WR', 'WS', 'WU', 'WV', 'WW', 'WX', 'WY', 'WZ', 'XA', 'XB', 'XC', 'XD', 'XE', 'XF', 'XG', 'XH', 'XI', 'XJ', 'XK', 'XL', 'XM', 'XN', 'XO', 'XP', 'XQ', 'XR', 'XS', 'XT', 'XU', 'XV', 'XW', 'XX', 'XY', 'XZ', 'YA', 'YB', 'YC', 'YD', 'YE', 'YF', 'YG', 'YH', 'YI', 'YJ', 'YK', 'YL', 'YM', 'YN', 'YO', 'YP', 'YQ', 'YR', 'YS', 'YT', 'YU', 'YV', 'YW', 'YX', 'YY', 'YZ', 'ZA', 'ZB', 'ZC', 'ZD', 'ZE', 'ZF', 'ZG', 'ZH', 'ZI', 'ZJ', 'ZK', 'ZL', 'ZM', 'ZN', 'ZO', 'ZP', 'ZQ', 'ZR', 'ZS', 'ZT', 'ZU', 'ZV', 'ZW', 'ZX', 'ZY', 'ZZ', 'R*', 'S*', 'E*', 'W*', 'Q*', 'Qs', 'Gs', 'Es', 'Ls', 'G*', 'Ns', 'Ds', 'Fs', 'C*', 'Ys', 'Ks', 'Ps', 'Y*', 'Vs', 'K*', 'Is', '1L', 'L*', 'Rs', 'Ts', 'As', 'Ss', 'Ws', 'Hs', '1I', 'Cs', '**', '3P', '3R', 'Ms', '4K', '4Y', '3L', '4A', '8Y', '4S', '-s', '3*', '5L', '7*', '7L', '7S', '2S', '4L', '8R', '2N', '1S', '1*', '3W', '5*', '1V', '9C', '1F', '5H', 'F*', '2K', '8I', '3F', 'T*', '1K', '5K', '3H', '1R', '2L', 'Rl', '2*', '5S', '6V', '6L', '8*', '1N', '2E', '3I', '8L', '*s', '9Y', '1C', '2T', '8K', '5C', '6C', '2Q', '2V', '5W', '5Y', '8S', '9P', 'I*', '1Y', '4N', '4*', '6*', '5F', '2R', '5M', '2C', '5I', 'V*', '4F', '1A', '3V', '7C', '4l', '8T', '9L', '4I', '4V', '3K', '8C', '2D', '8E', '7D', '9F', '1W', '4W', '2Y', '3S', '2l', '6Y', '9R', '2W', '7R', 'Ll', 'Nl', 'Kl', 'El', 'Tl', 'Gl', '4R', 'Sl', 'Fl', 'Il', 'Al', 'Vl', 'Cl', 'Hl', 'Pl', 'Ql', 'Yl', 'Dl', 'Ml', 'Wl', '7W', 'M1', '1l', '8l', '1H']
```

### Train & Save model
train set과 validation set을 나눴고, XGBoost를 사용해서 모델을 훈련 시켰다.

```python
X_train, X_val, y_train, y_val = train_test_split(X_prep, y_subclass, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=7,
    random_state=42,
    eval_metric='mlogloss',
    reg_lambda=15,
    subsample=0.5, 
    colsample_bytree=0.5
)

eval_set = [(X_val, y_val)]

model.fit(X_train, y_train, eval_set=eval_set, verbose=100)

model_filename = './model_saved/xgboost_model.bin'
model.save_model(model_filename)
print(f"모델이 {model_filename}에 저장되었습니다.")
```

#### 결과는 아래와 같다.
```python
[0]	validation_0-mlogloss:3.19918
[100]	validation_0-mlogloss:2.16103
[200]	validation_0-mlogloss:2.06376
[300]	validation_0-mlogloss:2.03879
[400]	validation_0-mlogloss:2.03872
[500]	validation_0-mlogloss:2.04382
[600]	validation_0-mlogloss:2.05375
[700]	validation_0-mlogloss:2.06549
[800]	validation_0-mlogloss:2.08133
[900]	validation_0-mlogloss:2.09596
[999]	validation_0-mlogloss:2.11034
```

하이퍼파라미터 튜닝 작업도 충분히 되어있지 않고, 고차원의 머신러닝 기법들을 적용하지 않은 터라 validation 성능조차 잘 나오지 않는다.

### Test data preprocessing & Test
테스트 데이터 역시 훈련용 데이터를 전처리 한 방식과 같은 방식으로 전처리를 해주고, 추론을 해서 결과파일을 저장하도록 했다.

```python
test = pd.read_csv("test.csv", header=0)

def preprocessing_test(element):
    # 띄어쓰기로 분리되어있는 element는 리스트로 쪼개기
    if isinstance(element, str) and ' ' in element:
        element = element.split()
    
    # 값이 한 개(문자열)인 경우
    if isinstance(element, str):
        if element[0] + element[-1] not in combination:
            combination.append(element[0] + element[-1])  ## 훈련용 데이터에서 없던 조합이 있을 수 있으므로 추가
        return np.where(element[0] == element[-1], 0, combination.index(element[0] + element[-1]))
        
    elif isinstance(element, list):
         return preprocessing_test(element[0])

test_X = test.drop(columns=['ID']).copy()
X_encoded_test = test_X.applymap(preprocessing_test)

predictions = model.predict(X_encoded_test)

original_labels = le_subclass.inverse_transform(predictions)

submission = pd.read_csv("sample_submission.csv")
submission["SUBCLASS"] = original_labels
submission.to_csv('./submission_dump/XGBoost_submission.csv', encoding='UTF-8-sig', index=False)
```

결과 파일을 데이콘 대회 사이트에 올려 성능을 보면 점수가 0.2점대 중반이 나왔다. 토크에서 봤듯이 외부 데이터 없이 많이들 내는 최선의 점수가 0.3점대 초반이라더니 머신러닝 기법을 적용했어도 명확한 한계가 있었을 것으로 보인다.

## 느낀점 / 후기 / 알게된 점
>시험 기간이라 시간도 부족했고, 도메인 지식도 부족한 상황에서 진행을 하니 결과가 좋지 않았던 것 같다. 하지만 이번 대회를 통해 도메인 지식의 중요성과 모델 구성의 전략이 상당히 중요하다는 점을 느꼈다. 이제와서 돌아보면 당연한 얘기지만 당시에는 막연히 부딪히고자 하는 마음이 들었었나보다. 하지만 대회가 종료되고, 순위가 높았던 팀들의 최종 발표 pdf 파일을 보니 진행하는 순서가 확실히 달랐다. \\
>시작은 대회 데이터를 분석한다. 단순히 구조를 파악하는 것이 아닌, 관련 도메인 지식을 철저히 공부하고 적용시켜 전처리 과정에 있어 체계적으로 전략을 세운다. 또한 데이터를 시각화해서 왜곡된 분포는 없는지, 구조는 어떠한지 등을 분석한다. 그 다음, 세운 전략을 바탕으로 다양한 전처리를 시도한다. 그 후 class 불균형이나 데이터의 부족의 경우는 외부데이터를 활용하여 같이 전처리를 하고, 전략적인 기법을 통해 학습을 진행한다. 그 후 모델을 구축하고 검증한다. \\
>이러한 체계적인 진행을 보고 정말 대단하다는 생각과 함께 나는 아직 배워야 할게 많고, 성장해야 한다는 느낌을 받았다. 다음 대회에서는 이러한 체게적인 방법을 바탕으로 더욱 진지하고 몰입적으로 참여할 것이다.