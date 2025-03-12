---
layout: single
title: "이미지 분류 해커톤: 데이터 속 아이콘의 종류를 맞춰라(작성 중)"
excerpt: "이미지 분류 대회 참가 기록"
mathjax: true
toc: true
toc_sticky: true
toc_label: Code
categories: Dacon
tag: [AI, ML, Competition, CV]
---


## 대회 주제
아이콘 이미지를 분류하는 AI 알고리즘을 개발하는 것이다.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cv2
from tqdm import tqdm
```

## Data Analysis
#### Data Shape
train.csv 파일 0열에는 ID, 1열에는 label, 2열 이후로는 이미지의 픽셀 데이터 정보가 들어가 있다.
test.csv 파일 0열에는 ID, 1열 이후로는 이미지의 픽셀 데이터 정보가 들어가 있다.
이미지 데이터는 32x32 Grayscale 이미지를 flatten 한 1D array 형태를 하고 있다.

#### Check Distribution of Train Data
```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_distribution = train['label'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(train_distribution.index, train_distribution.values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution in Train Set')
for i in range(len(train_distribution)):
    plt.text(i, train_distribution.values[i]+0.5, str(train_distribution.values[i]), ha='center')
plt.show()
```
![distribution1](../images/2025-03-11-image_classification/distribution1.png)
데이터 분포를 확인해봤더니 불균형이 심한 것을 알게 되었다. 불균형을 먼저 가장 큰 수인 128에 맞춰 다른 label의 데이터들을 증강시키기로 결정했다. 사실 이 단계는 증강이라고 하기에는 샘플링 해서 복사하는 것이기 때문에 불균형 해소 정도만 수행했다고 볼 수 있다.

```python
for lab in train['label']:
    tmp_df = train[train['label'] == lab]
    tmp_n = 128-len(tmp_df)
    if tmp_n > 64:
        train = pd.concat([train, tmp_df.sample(n=tmp_n, replace=True)])
    else:
        train = pd.concat([train, tmp_df.sample(n=tmp_n)])
train_distribution2 = train['label'].value_counts()
plt.figure(figsize=(10, 6))
plt.bar(train_distribution2.index, train_distribution2.values)
plt.xlabel('Label')
plt.ylabel('Count')
plt.title('Label Distribution in Train Set')
for i in range(len(train_distribution2)):
    plt.text(i, train_distribution2.values[i]+0.5, str(train_distribution2.values[i]), ha='center')
plt.show()
```
@ bar chart 이미지

#### Check Image Shape
```python
num_features = train.iloc[:, 2:].shape[1]
img_size = int(np.sqrt(num_features))
print(f"추정 이미지 크기: {img_size}x{img_size}")
```
일단 불균형을 해결했으니 데이터의 크기를 먼저 알아봐야 한다. 대회 홈페이지에서 주어지긴 했지만 나중에 주어지지 않은 상황에서의 능력을 기르기 위해 계산하는 부분도 짰다.
추정 이미지 크기: 32x32 가 나온다.

## Data Preprocessing
#### Seperate Target & Label Encoding
```python
X = train.iloc[:, 2:].values
y = train["label"].values
X_test = test.iloc[:, 1:].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```
먼저 훈련 타겟인 이미지 픽셀 데이터와 답지인 라벨을 각각 X 와 y 로 분리하고, 라벨은 라벨인코더를 통해 인코딩했다.

#### Data Augmentation
```python
def augment_images(X, y, img_size, n_augments=3):
    X_augmented = X.copy()
    y_augmented = y.copy()

    for i in tqdm(range(len(X))):
        img = X[i].reshape(img_size, img_size).astype(np.uint8)

        for j in range(n_augments):
            augmented_img = img.copy()

            # # 밝기 조정 (0.8-1.2 범위)
            # if np.random.random() > 0.5:
            #     brightness_factor = np.random.uniform(0.9, 1.1)
            #     augmented_img = np.clip(augmented_img * brightness_factor, 0, 255).astype(np.uint8)

            # 이미지 이동
            if np.random.random() > 0.5:
                tx, ty = np.random.randint(-3, 2, 2)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                augmented_img = cv2.warpAffine(augmented_img, M, (img_size, img_size))

            # 가우시안 노이즈 추가
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 10, (img_size, img_size)).astype(np.uint8)
                augmented_img = np.clip(augmented_img + noise, 0, 255).astype(np.uint8)

            # 증강된 이미지를 다시 1D로 변환
            X_augmented = np.vstack([X_augmented, augmented_img.flatten()])
            y_augmented = np.append(y_augmented, y[i])
    return X_augmented, y_augmented

print("데이터 증강 적용 중...")
X_augmented, y_augmented = augment_images(X, y, img_size, n_augments=11)
print(f"원본 데이터 크기: {len(X)}, 증강 후 데이터 크기: {len(X_augmented)}")


X_train, X_valid, y_train, y_valid = train_test_split(
    X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
)
```
데이터 증강 적용 중...
100%|██████████| 1280/1280 [03:54<00:00,  5.46it/s]원본 데이터 크기: 1280, 증강 후 데이터 크기: 14080

데이터 불균형을 해소했으니, 증강을 할 차례이다. 평행 이동, 가우시안 노이즈 추가 이 두 가지의 방법을 사용했다. 회전은 제출 시 점수가 떨어지는 것을 보고 제거했더니 오히려 점수가 올라서 코드에서 아예 지웠고, 밝기 조정은 강도를 약하게 하면 의미가 있을 것이라는 생각이 아직 남아있어서 주석처리만 해놓았다. 증강률(n_augmented)은 다양하게 시도를 해보았지만 현재 리더보드상 가장 높은 점수는 증강률을 11로 했을 때가 0.952 로 좋았다.


## Train
```python
model = XGBClassifier(
    n_estimators=20000,
    learning_rate=0.08,
    max_depth=9,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False,
    early_stopping_rounds=30,
    verbose_eval=20,
    reg_lambda=0.2,
    tree_method='gpu_hist' if torch.cuda.is_available() else 'hist'
)

# 모델 학습
print("모델 학습 중...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True
)
```
@ 학습 로그 텍스트 복붙

학습 시 XGBoost 모델을 사용했다. 초기에는 LightGBM과 함께 하이퍼파라미터를 조정하며 테스트를 했는데, LightGBM은 데이터 증강률이 높을 수록 속도는 조금 더 빠른데 Validation Accuracy가 상대적으로 조금 낮게 나와서 XGBoost 를 채택하였다.
증강률이 11이라 증강 후 총 데이터 수는 14080개이다. 그런데 n_estimator 값을 30000 으로 했을 때보다 20000 일때가 점수가 더 높았다. 사실상 약 1epoch이 감소하고 데이터들은 증강되어 있기에 오버피팅을 고려하기엔 모자르다고 생각이 들어 더 연구를 해봐야겠다. 다른 주요 하이퍼파라미터로는 모델의 학습률인 learning rate 는 0.08, 최대 트리의 깊이를 설정하는 max_depth 는 9, L2 정규화 파라미터인 reg_lambda 는 1.2로 설정했다. 25.03.11 기준 아직 scale_pos_weight 라는 파라미터는 사용하지 않았다. 데이터 불균형 시 label 가중치를 조절하는 파라미터인데 추후 사용해보고 비교해서 내용 추가하도록 하겠다.


## Validation
```python
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print("\nValidation Accuracy: ", valid_accuracy, '%', sep='')
```

@ 검증 결과 텍스트 복붙
위의 파라미터대로 검증 수행 시 Validation Accuracy는 이렇게 나온다.


## Test & Submission
```python
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(y_pred)

submission = pd.read_csv('sample_submission.csv')
submission['label'] = y_pred_labels
submission.to_csv('xgboost_augmented_submission.csv', index=False, encoding='utf-8-sig')
print("완료! 결과가 xgboost_augmented_submission.csv 파일에 저장되었습니다.")
```
완료! 결과가 xgboost_augmented_submission.csv 파일에 저장되었습니다.

제출 양식 파일(submission.csv) 파일을 읽고 양식에 맞춰 제출할 답안지를 만드는 코드이다.


## 돌아보기
#### 알게 된 점


#### 스스로에 대한 고찰


ㅡㅡㅡㅡ

![Leaderboard](../images/2025-03-11-image_classification/Leaderboard.png)
