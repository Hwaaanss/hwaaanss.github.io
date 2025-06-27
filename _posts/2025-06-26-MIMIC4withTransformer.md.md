---
layout: single
title: MIMIC-IV 데이터를 Transformer 를 활용하여 처치 및 시술 등의 순서 추천 모델 개발
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
- `subject_id` 환자 개개인을 식별하는 고유 ID이며, 다른 테이블과 정보를 연결하는 데 사용된다.
- `hadm_id` 병원 입원을 식별하는 고유 ID로, 이 모델 분석의 중심이 되는 기본 키이다.
- `admittime` 환자의 입원 시간 정보이며, 이벤트들의 시간 순서를 정렬하는 기준점으로 사용된다.
- `hospital_expire_flag` 환자의 입원 중 사망 여부를 0 또는 1로 나타내며, 우리 모델이 최종적으로 예측해야 할 목표이다.

#### patients.csv
- `subject_id` 환자 고유 ID이며, 입원 정보와 연결하는 데 사용된다.
- `gender` 환자의 성별 정보이며, 모델에 입력되는 고정 피처 중 하나로 토큰화된다.
- `anchor_age` 환자의 나이 정보이며, 10대 단위로 구간화되어 모델에 입력되는 고정 피처로 사용된다.

#### diagnoses_icd.csv
- `hadm_id` 어떤 입원 기록에 해당하는 진단인지 연결해주는 키이다.
- `icd_code` 국제질병분류 진단 코드를 의미하며, 환자의 초기 상태를 정의하는 핵심 정보로 토큰화된다.

#### prescriptions.csv
- `hadm_id` 어떤 입원 기록에 해당하는 처방인지 연결해주는 키이다.
- `starttime` 약물 투여 시작 시간이며, 처치 이벤트들의 순서를 결정하는 기준 시간으로 사용된다.
- `drug` 처방된 약물의 이름이며, 시간에 따라 변화하는 핵심 이벤트 정보로 토큰화되어 시퀀스를 구성한다.

#### procedures_icd.csv
- `hadm_id` 어떤 입원 기록에 해당하는 시술인지 연결해주는 키이다.
- `chartdate` 시술이 기록된 날짜이며, 다른 이벤트들과의 순서를 결정하는 기준 시간으로 사용된다.
- `icd_code` 국제질병분류 시술 코드를 의미하며, 시간에 따라 변화하는 핵심 이벤트 정보로 토큰화되어 시퀀스를 구성한다.

## Pipeline
#### Phase 1: 생존률 예측 모델 구축 (Foundation)
Phase 1 은 주어진 환자 정보와 이벤트 시퀀스에 대한 생존률을 정확하게 예측하는 Transformer 모델을 개발하는 것을 메인으로 진행된다.

**1단계: 코호트 정의 및 데이터 추출 (Cohort & Data Extraction)**
	MIMIC-IV에서 분석 대상 환자군(코호트)을 정의하고, 필요한 모든 테이블에서 데이터를 추출하여 결합.
        
**2단계: 피처 엔지니어링 및 통합 사전 구축 (Feature Engineering & Vocabulary)**
    환자의 고정 정보(나이, 몸무게, 성별, 진단명 등)와 시계열 이벤트(처방, 시술 등)를 전처리.
    모든 이벤트를 고유한 정수 토큰으로 매핑하는 통합 단어장(Vocabulary)을 생성.
        
**3단계: 환자별 시퀀스 생성 및 입력 데이터 구성 (Sequence Generation)**
	각 환자별로 [고정 정보 토큰] + [시계열 이벤트 토큰] 형태의 통합 시퀀스를 생성.
	모델 학습에 필요한 최종 입력 텐서(input_ids, attention_mask 등)를 구성.
        
**4단계: Transformer 예측 모델 아키텍처 설계 (Model Architecture)**  
	생성된 시퀀스를 입력받아 생존률(0~1 사이의 값)을 출력하는 Transformer 기반 모델을 PyTorch로 설계.
        
**5단계: 모델 학습 및 평가 (Training & Evaluation)**
    구성된 데이터셋으로 모델을 학습시키고, AUROC, AUPRC 등의 지표로 예측 성능을 검증.

#### Phase 2: 처치 순서 추천 시스템 구현 (Advanced Goal)
Phase 2는 예측 모델후 완성 후 이를 활용하여 처치 순서를 추천하도록 시스템 구현을 메인으로 진행한다.

**6단계: 추천 전략 수립 및 구현 (Recommendation Strategy)** 
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
import pandas as pd
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import math
import numpy as np
import random
```
```python
admissions = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/admissions.csv')
patients = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/patients.csv')
diagnoses = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/diagnoses_icd.csv')
prescriptions = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/prescriptions.csv')
procedures = pd.read_csv('/content/drive/MyDrive/Research/MIMIC4withTransformer/mimic-iv-3.1/hosp/procedures_icd.csv')
```
```python
df_base = pd.merge(admissions, patients, on='subject_id')[['hadm_id', 'admittime', 'anchor_age', 'gender', 'hospital_expire_flag']].copy()
df_base.rename(columns={'anchor_age': 'age'}, inplace=True)
df_base.dropna(inplace=True)
```

## Feature Engineering & Vocabulary
환자의 고정 정보(나이, 몸무게, 성별, 진단명 등)와 시계열 이벤트(처방, 시술 등)를 전처리한다. 모든 이벤트를 고유한 정수 토큰으로 매핑하는 통합 단어장(Vocabulary)을 생성한다. 해당 작업의 코드는 아래와 같다.
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
```python
all_event_tokens = events_df['event_value'].unique().tolist()

static_tokens = [f'AGE_{i}' for i in range(10)] + ['GENDER_M', 'GENDER_F']
special_tokens = ['[PAD]', '[CLS]', '[SEP]']


vocab_list = special_tokens + static_tokens + all_event_tokens
vocab = {token: i for i, token in enumerate(vocab_list)}
```

## Sequence Generation
각 환자별로 [고정 정보 토큰] + [시계열 이벤트 토큰] 형태의 통합 시퀀스를 생성하고, 모델 학습에 필요한 텐서형태로 구성한다. 전처리가 끝났으면 pkl 파일로 저장한다.
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
file_path = 'processed_data.pkl'
with open(file_path, 'wb') as f:
	pickle.dump(data_to_save, f)

print("-" * 50)
print(f"전처리 완료. '{file_path}' 파일 저장 성공.")
print(f"총 {len(all_patient_data)}명의 환자 데이터가 처리되었습니다.")
```

## Model Architecture
생성된 시퀀스를 입력받아 생존률(0~1 사이의 값)을 출력하는 Transformer 기반 모델을 PyTorch로 설계한다. 
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
		nn.Linear(64, 1), nn.Sigmoid())
	
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
```python
def train_model(model, train_loader, val_loader, num_epochs=10):
	model.to(device)

	optimizer = optim.AdamW(model.parameters(), lr=1e-4)
	criterion = nn.BCELoss()
	
	print("\n--- 모델 학습 시작 ---")
	
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
	
		val_loss, val_acc = 0, 0
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
		print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
	
	print("--- 모델 학습 완료 ---\n")
	return model
```
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

print("모델링 관련 클래스 및 함수 정의 완료.")
```