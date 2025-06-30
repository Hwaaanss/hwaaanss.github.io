---
layout: single
title: MIMIC-IV 데이터를 Transformer 를 활용하여 처치 및 시술 등의 순서 추천 모델 개발(작성 중)
excerpt: 치료의 조합보다 순서에 집중하는 모델
mathjax: true
toc: true
toc_sticky: true
toc_label: Code
categories: Research
tags:
  - DL
  - ML
  - Bio
---
## Idea
데이터베이스 강의에서 팀 프로젝트 과제로 MIMIC-IV 데이터를 활용해 데이터 분석하는 프로젝트가 있었는데, 치료라는 것이 "같은 조합의 처치이더라도 다른 순서로 받았을 때의 결과는 과연 같을까?" 라는 개인적인 아이디어에서 시작되어 따로 연구를 하게 되었다. ~~팀 프로젝트에는 Tabular 데이터 처리에 용이한 FT-Transformer 를 통해 사망률을 예측하는 것으로 아이디어를 내고, LLM 을 통해 MIMIC-IV 임상노트를 학습해 멀티모달로 결합하는 것으로 진행했다.~~ 

## Details
연구 정의: 환자 정보(나이, 몸무게, 성별, 진단명 등)와 시계열적 이벤트 정보(순서가 중요)를 보고 처치 순서 추천 및 생존률 예측.

사용 데이터: MIMIC-IV

데이터 전처리 포인트: 처치 정보 사용에 제한을 두지 않고, 모든 기간의 정보를 활용. 시간 정보는 한 환자에 대해 데이터를 구성할 때, 처치/투여/시술 등의 이벤트들의 순서를 맞추기 위해서만 사용됨.

사용 모델: Transformer, 커스텀 Survival Predictor 

모델의 주요 학습 포인트: 환자 정보(나이, 몸무게, 성별, 진단명)와 처치/투여/시술 등의 순서가 갖는 생존 여부를 학습.

모델의 최종 목표: 현재 이 환자에게 권장되는 순서의 처치나 투여를 추천해주고, 이 순서를 따라 치료를 진행했을 때에 예측되는 생존률(or 사망률)은 몇 퍼센트인지 출력.

## Data EDA
#### admissions.csv
- `subject_id`: 환자 개개인을 식별하는 고유 ID이며, 다른 테이블과 정보를 연결하는 데 사용된다.
- `hadm_id`: 병원 입원을 식별하는 고유 ID로, 이 모델 분석의 중심이 되는 기본 키이다.
- `admittime`: 환자의 입원 시간 정보이며, 이벤트들의 시간 순서를 정렬하는 기준점으로 사용된다.
- `hospital_expire_flag`: 환자의 입원 중 사망 여부를 0 또는 1로 나타내며, 우리 모델이 최종적으로 예측해야 할 목표이다.

#### patients.csv
- `subject_id`: 환자 고유 ID이며, 입원 정보와 연결하는 데 사용된다.
- `gender`: 환자의 성별 정보이며, 모델에 입력되는 고정 피처 중 하나로 토큰화된다.
- `anchor_age`: 환자의 나이 정보이며, 10대 단위로 구간화되어 모델에 입력되는 고정 피처로 사용된다.

#### diagnoses_icd.csv
- `hadm_id`: 어떤 입원 기록에 해당하는 진단인지 연결해주는 키이다.
- `icd_code`: 국제질병분류 진단 코드를 의미하며, 환자의 초기 상태를 정의하는 핵심 정보로 토큰화된다.

#### prescriptions.csv
- `hadm_id`: 어떤 입원 기록에 해당하는 처방인지 연결해주는 키이다.
- `starttime`: 약물 투여 시작 시간이며, 처치 이벤트들의 순서를 결정하는 기준 시간으로 사용된다.
- `drug`: 처방된 약물의 이름이며, 시간에 따라 변화하는 핵심 이벤트 정보로 토큰화되어 시퀀스를 구성한다.

#### procedures_icd.csv
- `hadm_id`: 어떤 입원 기록에 해당하는 시술인지 연결해주는 키이다.
- `chartdate`: 시술이 기록된 날짜이며, 다른 이벤트들과의 순서를 결정하는 기준 시간으로 사용된다.
- `icd_code`: 국제질병분류 시술 코드를 의미하며, 시간에 따라 변화하는 핵심 이벤트 정보로 토큰화되어 시퀀스를 구성한다.

## Pipeline
#### Phase 1: 생존률 예측 모델 구축 (Foundation)
Phase 1 은 주어진 환자 정보와 이벤트 시퀀스에 대한 생존률을 정확하게 예측하는 Transformer 모델을 개발하는 것을 메인으로 진행된다.

1. 코호트 정의 및 데이터 추출 (Cohort & Data Extraction)
	MIMIC-IV에서 분석 대상 환자군(코호트)을 정의하고, 필요한 모든 테이블에서 데이터를 추출하여 결합.
        
2. 피처 엔지니어링 및 통합 사전 구축 (Feature Engineering & Vocabulary)
    환자의 고정 정보(나이, 몸무게, 성별, 진단명 등)와 시계열 이벤트(처방, 시술 등)를 전처리.
    모든 이벤트를 고유한 정수 토큰으로 매핑하는 통합 단어장(Vocabulary)을 생성.
        
3. 환자별 시퀀스 생성 및 입력 데이터 구성 (Sequence Generation)
	각 환자별로 [고정 정보 토큰] + [시계열 이벤트 토큰] 형태의 통합 시퀀스를 생성.
	모델 학습에 필요한 최종 입력 텐서(input_ids, attention_mask 등)를 구성.
        
4. Transformer 예측 모델 아키텍처 설계 (Model Architecture)
	생성된 시퀀스를 입력받아 생존률(0~1 사이의 값)을 출력하는 Transformer 기반 모델 설계.
        
5. 모델 학습 및 평가 (Training & Evaluation)
    구성된 데이터셋으로 모델을 학습시키고, AUROC, AUPRC 등의 지표로 예측 성능을 검증.

#### Phase 2: 처치 순서 추천 시스템 구현 (Advanced Goal)
Phase 2는 예측 모델후 완성 후 이를 활용하여 처치 순서를 추천하도록 시스템 구현을 메인으로 진행한다.

6.  추천 전략 수립 및 구현 (Recommendation Strategy)
	단순히 학습된 모델로 한 번 예측하고 끝나는 것이 아니라, 이 모델을 일종의 시뮬레이터로 활용하는 것이 이 단계의 핵심이다. 즉, 환자의 현재 상태를 기반으로 앞으로 가능한 모든 처치들을 가상으로 적용해보며 어떤 경로가 가장 높은 생존률로 이어지는지를 탐색하는 방식이다.
	이때, 매 단계에서 가장 좋아 보이는 단 하나의 선택지만을 따르는 탐욕적(Greedy) 접근법은 단기적인 최적해에 갇힐 위험이 있다. 이를 보완하기 위해, 가장 유망한 상위 몇 개의 경로(beam)를 동시에 고려하며 탐색을 이어나가는 빔 서치(Beam Search) 전략을 채택했다.

빔 서치의 과정은 다음과 같다.

1. **초기 상태 설정:** 추천을 시작할 환자의 고정 정보(나이, 성별 등)와 초기 진단명으로 기본 시퀀스를 구성한다.
    
2. **경로 확장 및 평가:** 현재의 유망한 경로들 각각에 대해, 가능한 모든 처치/시술 후보군을 하나씩 추가하여 확장된 가상 경로들을 생성한다. 그리고 Phase 1에서 학습된 커스텀 `SurvivalPredictor` 모델을 통해 각 가상 경로의 예측 생존률을 평가한다.
    
3. **경로 선택:** 평가된 생존률을 기준으로 가장 점수가 높은 상위 N개(Beam Width)의 경로만을 남기고 나머지는 폐기한다.
    
4. **반복:** 원하는 길이의 처치 시퀀스가 생성될 때까지 2, 3번 과정을 반복한다.

이 과정을 통해 최종적으로 가장 높은 생존 확률을 보인 시퀀스를 추천 처치 순서로 제시하고, 해당 확률 값을 함께 출력하여 의사결정을 지원하는 것을 목표로 한다.

## Cohort & Data Extraction
위 Details 에서 언급한데로 대상 환자는 성인이라면 별도의 제한을 두지 않고 모두 학습에 활용하기로 했다. 데이터를 불러오고 병합하는 작업의 코드는 아래와 같다. 
```python
import os
from datetime import datetime
import math
import random
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
```
```python
admissions = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/admissions.csv')
patients = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/patients.csv')
diagnoses = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/diagnoses_icd.csv')
prescriptions = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/prescriptions.csv')
procedures = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/procedures_icd.csv')
```

## Data Preprocessing
환자의 고정 정보(나이, 몸무게, 성별, 진단명 등)와 시계열 이벤트(처방, 시술 등)를 전처리한다. 모든 이벤트를 고유한 정수 토큰으로 매핑하는 통합 단어장을 생성한다.
#### Merge Data
subject_id 을 key로 잡고 환자 기본정보들 병합.
- 기본정보: 환자 고유 ID, 입원 고유 ID, 입원 시각, 나이, 성별, 사망여부
```python
df_base = pd.merge(admissions, patients, on='subject_id')[['hadm_id', 'admittime', 'anchor_age', 'gender', 'hospital_expire_flag']].copy()
df_base.rename(columns={'anchor_age': 'age'}, inplace=True)
df_base.dropna(inplace=True)
```

#### Gather Sequential Data
- diagnoses: 입원 고유 ID를 기준으로 left outer join 실행. event_type 컬럼을 만들고, 값은 DIAG 로 채움. event_value 컬럼을 만들고, 값은 DIAG_ + 진단코드로 채움.
- prescriptions: 처방 관련 정보들을 위의 diagnoses 에 한 작업과 같이 전처리 수행
- procedures: 처치/시술 관련 정보들을 위의 diagnoses 에 한 작업과 같이 전처리 수행
- 이후 전처리 한 컬럼들을 concat 하고 정렬해서 별도 dataframe 으로 저장

```python
admittime_df = df_base[['hadm_id', 'admittime']]

diagnoses = pd.merge(diagnoses, admittime_df, on='hadm_id', how='left')
diagnoses['event_type'] = 'DIAG'
diagnoses['event_value'] = 'DIAG_' + diagnoses['icd_code'].astype(str)
diagnoses['time'] = pd.to_datetime(diagnoses['admittime'])


prescriptions['event_type'] = 'MED'
prescriptions['event_value'] = 'MED_' + prescriptions['drug'].astype(str)
prescriptions['time'] = pd.to_datetime(prescriptions['starttime'])


procedures['event_type'] = 'PROC'
procedures['event_value'] = 'PROC_' + procedures['icd_code'].astype(str)
procedures['time'] = pd.to_datetime(procedures['chartdate'])


events_df = pd.concat([
diagnoses[['hadm_id', 'time', 'event_type', 'event_value']], prescriptions[['hadm_id', 'time', 'event_type', 'event_value']], procedures[['hadm_id', 'time', 'event_type', 'event_value']]])

events_df.dropna(subset=['hadm_id', 'time', 'event_value'], inplace=True)
events_df.sort_values(by=['hadm_id', 'time'], inplace=True)
```

#### Make Intergrated Dictionary
- all_event_tokens: 처치/시술/처방/진단 관련 모든 종류의 값을 포함한 리스트
- static_tokens: 연령대를 10단위로 구분한 리스트와 성별을 포함한 리스트
- special_tokens: transformer가 시퀀스 데이터를 처리하기 위한 토큰들로, 길이가 짧은 부분을 padding 처리하는 토큰인 PAD, 시퀀스의 처음에 들어가는 토큰인 CLS, 환자 기본정보와 처치/시술 등의 정보를 구분하기 위한 토큰인 SEP 으로 구성
- vocab_list: 위의 모든 리스트들을 합친 리스트
- vocab: vocab_list 값들과 인덱스들로 만든 dictionary
```python
all_event_tokens = events_df['event_value'].unique().tolist()

static_tokens = [f'AGE_{i}' for i in range(10)] + ['GENDER_M', 'GENDER_F']
special_tokens = ['[PAD]', '[CLS]', '[SEP]']


vocab_list = special_tokens + static_tokens + all_event_tokens
vocab = {token: i for i, token in enumerate(vocab_list)}
```

#### Sequence Generation
각 환자별로 [고정 정보 토큰] + [시계열 이벤트 토큰] 형태의 통합 시퀀스를 생성하고, 모델 학습에 필요한 텐서형태로 구성한다. 
- age_token: 연령대(10자리)를 토큰화한 컬럼으로, 90대 이상은 모두 동일하게 9로 처리
- gender_token: 성별을 토큰화
- static_info: 입원 고유 ID 와 토큰, 사망여부 컬럼들만 모은 dataframe. 즉 환자 기본 정보를 토큰화 한 dataframe
- diag_events: 진단 정보만 모음
- other_events: 진단 정보를 제외한 데이터만 모음
- diag_sequences: 입원 고유 ID 별로 진단 정보들의 event_value 들을 리스트화 한 것
- other_sequences: 입원 고유 ID 별로 진단 정보를 제외한 데이터들의 event_value 들을 리스트화 한 것(처치/시술/처방)
- final_df: 환자 기본 정보와 진단 정보/비 진단 정보들을 모두 left outer join 으로 병합한 것
```python
df_base['age_token'] = 'AGE_' + (df_base['age'] // 10).clip(upper=9).astype(str)
df_base['gender_token'] = 'GENDER_' + df_base['gender']


static_info = df_base[['hadm_id', 'age_token', 'gender_token', 'hospital_expire_flag']]


diag_events = events_df[events_df['event_type'] == 'DIAG']
other_events = events_df[events_df['event_type'] != 'DIAG']


diag_sequences = diag_events.groupby('hadm_id')['event_value'].agg(list).rename('diag_list')
other_sequences = other_events.groupby('hadm_id')['event_value'].agg(list).rename('other_list')


final_df = static_info.merge(diag_sequences, on='hadm_id', how='left')
final_df = final_df.merge(other_sequences, on='hadm_id', how='left')
final_df['diag_list'] = final_df['diag_list'].apply(lambda x: x if isinstance(x, list) else [])
final_df['other_list'] = final_df['other_list'].apply(lambda x: x if isinstance(x, list) else [])
```

#### Save Preprocessed pkl Data
- static_ids, diag_ids, other_ids: 결측값 공리스트로 저장했던거 제거
- input_ids: transformer 모델의 input 값으로써 필요한 토큰 처리 \
매번 전처리부터 할 수 없으므로 전처리한 데이터를 pkl 파일로 저장
```python
def convert_to_final_format(row, vocab):
	static_ids = [vocab.get(row['age_token']), vocab.get(row['gender_token'])]
	diag_ids = [vocab.get(t) for t in row['diag_list']]
	other_ids = [vocab.get(t) for t in row['other_list']]
	
	static_ids = [i for i in static_ids if i is not None]
	diag_ids = [i for i in diag_ids if i is not None]
	other_ids = [i for i in other_ids if i is not None]
	
	input_ids = [vocab['[CLS]']] + static_ids + diag_ids + [vocab['[SEP]']] + other_ids
	return {'input_ids': input_ids, 'label': row['hospital_expire_flag']}


all_patient_data = final_df.apply(lambda row: convert_to_final_format(row, vocab), axis=1).tolist()

data_to_save = {'all_patient_data': all_patient_data, 'vocab': vocab}
file_path = '/content/drive/MyDrive/Research/MIMIC4withTransformer/dump/processed_data.pkl'
with open(file_path, 'wb') as f:
	pickle.dump(data_to_save, f)

print("-" * 50)
print(f"전처리 완료. '{file_path}' 파일 저장 성공.")
print(f"총 {len(all_patient_data)}명의 환자 데이터가 처리되었습니다.")
```

## Define Model, DataLoader
생성된 시퀀스를 입력받아 생존률(0~1 사이의 값)을 출력하는 Transformer 기반 모델을 PyTorch로 설계.
#### PositionalEncoding Class
- `position`: 0부터 max_len-1까지의 순서(위치)를 나타내는 텐서를 생성 (ex. [[0], [1], [2], ...])
- `div_term`: Positional Encoding 수식에 사용될 분모 부분을 미리 계산해서 저장. 사인(sin)과 코사인(cos) 함수의 주기를 조절하여 위치별로 고유한 값을 만들어내는 핵심적인 부분
- `pe`: 최종적인 위치 인코딩 값을 저장할 비어있는 텐서를 생성
- `pe[:, 0, 0::2]`: 위치 인코딩 텐서의 짝수 인덱스에 사인(sin) 함수를 적용한 값을 채워 넣음
- `pe[:, 0, 1::2]`: 위치 인코딩 텐서의 홀수 인덱스에 코사인(cos) 함수를 적용한 값을 채워 넣음
- `self.register_buffer('pe', pe)`: pe 텐서를 모델의 파라미터(가중치)로는 취급하지 않지만, 모델의 상태로는 저장하는 버퍼로 등록. 이렇게 하면 모델 저장 및 로드 시 pe 값이 함께 관리e된다.(디버깅 중 gemini 가 마음대로 넣은 기능인데 괜찮아서 일단 보류)
- `x = x + self.pe[:x.size(0)]`: 모델에 입력된 단어 임베딩(x)에 미리 계산해둔 위치 인코딩(pe) 값을 더해서 모델은 단어의 의미뿐만 아니라 순서 정보도 함께 학습하게 된다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):
	def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
		super().__init__()		
		self.dropout = nn.Dropout(p=dropout)
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_len, 1, d_model)
		pe[:, 0, 0::2] = torch.sin(position * div_term)
		pe[:, 0, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.pe[:x.size(0)]
		return self.dropout(x)
```

#### SurvivalPredictor Class
생존율을 예측하는 모델 정의
- `self.d_model`: 모델 전체에서 사용될 임베딩 및 피처의 기본 차원 크기를 저장함.
- `self.embedding`: 입력된 토큰 ID들을 d_model 차원의 벡터로 변환하는 임베딩 레이어
- `self.pos_encoder`: 단어의 순서 정보를 더해주기 위해 이전에 정의한 PositionalEncoding 클래스를 가져와 사용
- `encoder_layer`: Transformer의 핵심 블록인 인코더 레이어를 한 층 정의
- `self.transformer_encoder`: 정의된 encoder_layer를 num_encoder_layers 개수만큼 쌓아 전체 Transformer 인코더를 구성함.
- `self.classifier`: 최종적으로 생존률을 예측하기 위한 classifier 임. Transformer를 통과한 결과값을 입력받아 0과 1 사이의 확률 값으로 출력하는 작은 신경망
- `forward(self, src, src_padding_mask)`: 모델의 순전파(forward pass) 로직을 정의하는 함수임. 데이터가 각 레이어를 통과하는 순서를 결정함.
- `self.embedding(src) * math.sqrt(self.d_model)`: 입력 토큰(src)을 임베딩 벡터로 변환하고, Transformer 논문에 따라 d_model의 제곱근을 곱해 스케일링함.
- `src.transpose(0, 1)`: PositionalEncoding 레이어는 (시퀀스 길이, 배치 크기, 피처) 순서의 입력을 받으므로 차원을 일시적으로 바꿔줌.
- `self.pos_encoder(src)`: 위치 인코딩을 임베딩 벡터에 더해줌.
- `src.transpose(0, 1)`: 다음 Transformer 인코더 레이어에 입력하기 위해 차원 순서를 다시 (배치 크기, 시퀀스 길이, 피처)로 원복시킴.
- `output`: 최종 입력값을 Transformer 인코더에 통과시킴. src_key_padding_mask를 통해 [PAD] 토큰은 계산에서 무시하도록 함.
- `cls_output`: Transformer를 통과한 결과값 중, 가장 맨 앞에 위치한 [CLS] 토큰에 해당하는 출력 벡터만 추출함. 이 벡터는 전체 시퀀스의 의미를 요약, 압축한 대표값으로 사용됨.
- `survival_prob`: 추출된 [CLS] 토큰의 출력 벡터를 최종 분류기에 넣어 0과 1 사이의 생존 확률 값을 계산함.
- `return survival_prob`: 계산된 최종 생존 확률을 반환
```python
class SurvivalPredictor(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, src, src_padding_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_padding_mask)
        cls_output = output[:, 0]
        survival_prob = self.classifier(cls_output)
        return survival_prob
```

#### PatientSequenceDataset Class
전처리된 환자 데이터를 객체로 저장하는 클래스
#### collate_fn
- `batch.sort(~)`: 배치 내의 데이터들을 시퀀스 길이 순서대로 (긴 것부터 짧은 것 순으로) 정렬함. 이는 파이토치의 패딩 함수 효율을 높이기 위한 작업
- `sequences, labels`: 묶여있는 데이터에서 input_ids 시퀀스들과 label들을 각각 분리하여 별개의 리스트로 만듦.
- `padded_sequences`: 패딩(Padding)을 수행하는 부분. 한 배치 내에서 가장 긴 시퀀스를 기준으로, 나머지 짧은 시퀀스들의 뒷부분에 [PAD] 토큰을 추가하여 모든 시퀀스의 길이를 동일하게 맞춤.
- `attention_masks`: 패딩된 부분([PAD] 토큰)은 True, 실제 데이터는 False인 마스크를 생성함. 이 마스크는 Transformer가 의미 없는 패딩 부분에는 attention 하지 않도록 알려주는 역할
- `labels`: 분리되어 있던 개별 label 텐서들을 하나의 텐서로 쌓아줌.
```python
class PatientSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]['input_ids']), torch.tensor(float(self.data[idx]['label']))


def collate_fn(batch, vocab):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, labels = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=vocab['[PAD]'])
    attention_masks = (padded_sequences == vocab['[PAD]'])
    labels = torch.stack(labels)
    return padded_sequences, attention_masks, labels
```

#### train_model
딥러닝응용1 강의에서 배운 모델 정의 형식을 기반으로 수정함.
- `optimizer`: AdamW
- `criterion`: BCELoss
```python
def train_model(model, train_loader, val_loader, lr=1e-4, num_epochs=10):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    print("\n--- Start Training ---")
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for seq, mask, labels in train_loader:
            seq, mask, labels = seq.to(device), mask.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(seq, mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for seq, mask, labels in val_loader:
                seq, mask, labels = seq.to(device), mask.to(device), labels.to(device).unsqueeze(1)
                outputs = model(seq, mask)
                val_loss += criterion(outputs, labels).item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_auc = roc_auc_score(all_labels, all_preds)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")

    print("--- Finish Training ---\n")
    return model, history
```


#### recommend_and_predict
아래는 처치 순서 추천과 생존율 예측 프로세스를 위한 함수를 정의한 코드이다.
- `id2token`: 정수 ID를 다시 텍스트 토큰으로 변환하기 위한 역-사전을 생성
- `candidate_treatments`: 전체 사전(vocab)에서 MED_ 또는 PROC_로 시작하는 모든 처치/시술 토큰들을 골라 추천 후보 리스트를 만듦.
- `if not candidate_treatments~`: 만약 추천할 처치 후보가 하나도 없으면 오류 메시지와 함께 함수를 종료하는 안전장치
- `age_token`,`gender_token`: 입력받은 환자 정보(나이, 성별)를 이전에 정의한 규칙에 따라 토큰 텍스트로 변환함
- `static_ids`, `diag_ids`: 변환된 텍스트 토큰들을 사전(vocab)을 이용해 해당하는 정수 ID로 바꿈
- `initial_ids`: 환자의 고정 정보와 진단 정보를 합쳐 [CLS]와 [SEP] 토큰을 포함한 초기 시퀀스를 구성함. 이 시퀀스가 빔 서치의 시작점이 됨
- `beams`: 빔 서치를 시작하기 위한 초기 빔을 설정함. (시퀀스, 점수) 형태의 튜플을 리스트에 담아두며, 처음에는 초기 시퀀스와 점수 0으로 시작
- `all_candidates`: 현재 단계에서 생성될 수 있는 모든 후보 경로들을 임시로 저장할 리스트
- `for seq, score in beams`: 이전 단계에서 살아남은 유망한 경로(빔)들을 하나씩 꺼내옴.
- `for treatment in candidate_treatments`: 각 유망 경로 뒤에 가능한 모든 처치 후보들을 하나씩 붙여 새로운 가상 경로(new_seq) 만듦.
- `seq_tensor`, `mask`: 생성된 가상 경로를 모델에 입력하기 위해 텐서 형태로 변환하고, 패딩이 없으므로 모든 값이 False인 마스크를 생성
- `survival_prob`: 생성된 가상 경로를 모델에 입력하여 예측 생존 확률을 계산
- `all_candidates.append(~)`: 생성된 가상 경로와 그 경로의 생존 확률 점수를 all_candidates 리스트에 추가
- `ordered`: 현재 단계에서 생성된 모든 후보 경로들을 생존 확률 점수가 높은 순서대로 정렬
- `beams`: 정렬된 후보들 중 가장 점수가 높은 상위 N개(beam_width)만 남기고 나머지는 버림. 이 N개가 다음 단계의 빔 서치 대상이 됨.
- `if not beams: break`: 만약 살아남은 경로가 하나도 없으면 탐색을 중단
- `best_sequence, best_survival_rate`: 모든 탐색이 끝난 후, 최종적으로 남은 빔 중에서 가장 점수가 높은 첫 번째 경로를 최적의 결과로 선택
- `sep_index`, `recommended_ids`: 최적의 시퀀스에서 [SEP] 토큰의 위치를 찾아 그 이후의 처치에 해당하는 ID들만 추출
- `recommended_treatments`: 추출된 처치 ID들을 id2token 사전을 이용해 다시 디코딩
```python
def recommend_and_predict(patient_info, model, vocab, max_steps=5, beam_width=3):
    model.eval()
    
    id2token = {i: token for token, i in vocab.items()}
    candidate_treatments = [token for token in vocab.keys() if token.startswith('MED_') or token.startswith('PROC_')]
    if not candidate_treatments: 
	    return "오류: 처치/시술 후보군이 없습니다.", 0.0
	    
    age_token = f"AGE_{min(patient_info['age'] // 10, 9)}"
    gender_token = f"GENDER_{patient_info['gender']}"
    static_ids = [vocab[t] for t in [age_token, gender_token] if t in vocab]
    diag_ids = [vocab[d] for d in patient_info['diagnoses'] if d in vocab]
    initial_ids = [vocab['[CLS]']] + static_ids + diag_ids + [vocab['[SEP]']]
    beams = [(initial_ids, 0.0)]

    for step in range(max_steps):
        all_candidates = []
        for seq, score in beams:
            for treatment in candidate_treatments:
                new_seq = seq + [vocab[treatment]]
                seq_tensor = torch.tensor([new_seq], device=device)
                mask = torch.zeros_like(seq_tensor, dtype=torch.bool, device=device)
                with torch.no_grad():
                    survival_prob = model(seq_tensor, mask).item()
                all_candidates.append((new_seq, survival_prob))
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beams = ordered[:beam_width]
        if not beams: break

    best_sequence, best_survival_rate = beams[0]
    sep_index = best_sequence.index(vocab['[SEP]'])
    recommended_ids = best_sequence[sep_index+1:]
    recommended_treatments = [id2token[id] for id in recommended_ids]
    return recommended_treatments, best_survival_rate
```

## Train & Validation
구성된 데이터셋으로 모델을 학습시키고, 예측 성능을 검증한다. 
```python
print('Start Training Session')
print("1. Load Data...")
file_path = '/content/drive/MyDrive/Research/MIMIC4withTransformer/dump/processed_data.pkl'
try:
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    all_patient_data = loaded_data['all_patient_data']
    vocab = loaded_data['vocab']
    print(f"Finish load data at dir '{file_path}'\n")
except FileNotFoundError:
    print(f"Error: '{file_path}' Cannot found a file.")


print("2. Prepare Dataset and DataLoader...")
train_val_data, test_data = train_test_split(all_patient_data, test_size=0.1, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=42)

train_dataset = PatientSequenceDataset(train_data)
val_dataset = PatientSequenceDataset(val_data)
test_dataset = PatientSequenceDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda b: collate_fn(b, vocab))
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=lambda b: collate_fn(b, vocab))
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=lambda b: collate_fn(b, vocab))
print("Finish Loading.\n")


print("3. Reset Model and Start to Training...")
learning_rate = 1e-5
epochs = 20
model = SurvivalPredictor(vocab_size=len(vocab)).to(device)
trained_model, history = train_model(model, train_loader, val_loader, lr=learning_rate, num_epochs=epochs)


print("\n4. Visualize and Save Training History...")
fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(history['train_loss'], 'b-', label='Train Loss')
ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params('y', colors='b')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(history['val_auc'], 'g-s', label='Validation AUC')
ax2.set_ylabel('AUC', color='g')
ax2.tick_params('y', colors='g')
ax2.legend(loc='upper right')
fig.tight_layout()
plt.title('Model Training History')

plot_dir = 'training_plot'
os.makedirs(plot_dir, exist_ok=True)
plot_filename = f'{plot_dir}/{learning_rate}_{epochs}.png'
plt.savefig(plot_filename)
plt.show()
print(f"Training plot saved to '{plot_filename}'")


print("\n5. Recommend Treatments and Predict Survival ratio Simulation...")
id2token = {i: token for token, i in vocab.items()}

sep_token_id = vocab['[SEP]']
test_data_with_treatments = [
    p for p in test_data if len(p['input_ids']) > p['input_ids'].index(sep_token_id) + 1
]

if test_data_with_treatments:
    random_test_patient = random.choice(test_data_with_treatments)
else:
    print("경고: Test 셋에 처치 기록이 있는 환자가 없습니다. 임의의 환자를 사용합니다.")
    random_test_patient = random.choice(test_data)

test_patient_ids = random_test_patient['input_ids']

sep_index = test_patient_ids.index(vocab['[SEP]'])
cls_index = test_patient_ids.index(vocab['[CLS]'])

static_and_diag_ids = test_patient_ids[cls_index+1:sep_index]
static_and_diag_tokens = [id2token.get(id, '[UNK]') for id in static_and_diag_ids]

age_token = [t for t in static_and_diag_tokens if t.startswith('AGE_')][0]
gender_token = [t for t in static_and_diag_tokens if t.startswith('GENDER_')][0]
diag_tokens = [t for t in static_and_diag_tokens if t.startswith('DIAG_')]

age = int(age_token.split('_')[1]) * 10
gender = gender_token.split('_')[1]

new_patient_info = {
    'age': age,
    'gender': gender,
    'diagnoses': diag_tokens
}

recommended_sequence, predicted_survival_rate = recommend_and_predict(
    patient_info=new_patient_info,
    model=trained_model,
    vocab=vocab,
    max_steps=4,
    beam_width=3
)

actual_treatment_ids = test_patient_ids[sep_index+1:]
actual_treatment_sequence = [id2token.get(id, '[UNK]') for id in actual_treatment_ids]

print("-" * 50)
print("              < 최종 예측 결과 >")
print("-" * 50)
print(f"추출된 Test 환자 정보: {new_patient_info}")
print("\n▶ 추천되는 처치 순서:")
for i, treatment in enumerate(recommended_sequence):
    print(f"  {i+1} 단계: {treatment}")
print("\n▶ 예측 결과:")
print(f"  위 순서로 치료 시 예측되는 생존률: {predicted_survival_rate * 100:.2f}%")

print("-" * 50)
print("\n▶ 실제 처치 순서 (상위 4개):")
if actual_treatment_sequence:
    for i, treatment in enumerate(actual_treatment_sequence[:4]):
        print(f"  {i+1} 단계: {treatment}")
else:
    print("  기록된 실제 처치(MED/PROC) 내역이 없습니다.")
print(f"해당 환자의 실제 생존 여부: {'생존' if random_test_patient['label'] == 0 else '사망'}")
```

검토 사항: 추천한 시술들이 적합한지 의료전문가에게 추천 및 본 모델은 단지 "추천을 위한 모델"임을 강조.
