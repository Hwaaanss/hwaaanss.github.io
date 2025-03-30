---
layout: single
title: "이미지 분류 해커톤: 데이터 속 아이콘의 종류를 맞춰라"
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

## EDA(Exploratory Data Analysis)
#### Data Shape
train.csv 파일 0열에는 ID, 1열에는 label, 2열 이후로는 이미지의 픽셀 데이터 정보가 들어가 있다.
test.csv 파일 0열에는 ID, 1열 이후로는 이미지의 픽셀 데이터 정보가 들어가 있다.
이미지 데이터는 32x32 Grayscale 이미지를 flatten 한 1D array 형태를 하고 있다.(데이콘 제공)

#### Image appearance
sadas

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
![distribution](/images/2025-03-11-image_classification/distribution.png)
데이터 분포를 확인해봤더니 불균형이 심한 것을 알게 되었다. 불균형을 먼저 가장 큰 수인 128에 맞춰 다른 label의 데이터들을 증강시키기로 결정했다. 사실 이 단계는 증강이라고 하기에는 샘플링 해서 복사하는 것이기 때문에 불균형 해소 정도만 수행했다고 볼 수 있다.


## Data Preprocessing
#### Crop Images of Rabbit and Cat
```python
num_features = train.iloc[:, 2:].shape[1]
img_size = int(np.sqrt(num_features))
```
이미지의 사이즈를 측정해 변수로 저장해둔다.

```python
cat_rabbit_df = train[train['label'].isin(['cat', 'rabbit'])]
original_cat_rabbit = cat_rabbit_df.copy()

cat_count = len(train[train['label'] == 'cat'])
rabbit_count = len(train[train['label'] == 'rabbit'])
cat_augment_count = int(cat_count * 0.1)
rabbit_augment_count = int(rabbit_count * 0.1)
```
증강할 토끼와 고양이의 이미지 데이터셋을 추출하고, 각각의 증강 비율(개수)을 설정해준다.

```python
cat_to_augment = train[train['label'] == 'cat'].sample(n=cat_augment_count, random_state=42)
rabbit_to_augment = train[train['label'] == 'rabbit'].sample(n=rabbit_augment_count, random_state=42)
to_augment = pd.concat([cat_to_augment, rabbit_to_augment])
print(f"고양이 이미지 {cat_augment_count}개, 토끼 이미지 {rabbit_augment_count}개를 증강합니다.")
```
확대 및 크롭 작업을 할 이미지를 샘플링한다.

```python
augmented_rows = []
for idx, row in tqdm(to_augment.iterrows(), total=len(to_augment), desc="고양이/토끼 이미지 증강"):
    img_data = row.iloc[2:].values
    img = img_data.reshape(img_size, img_size).astype(np.float32)

    zoom_factor = 1.3
    zoomed_size = int(img_size * zoom_factor)
    zoomed_img = cv2.resize(img, (zoomed_size, zoomed_size), interpolation=cv2.INTER_LINEAR)

    crop_start = (zoomed_size - img_size) // 2
    crop_end = crop_start + img_size
    cropped_img = zoomed_img[crop_start:crop_end, crop_start:crop_end]

    augmented_img_data = cropped_img.flatten()

    new_row = row.copy()
    new_row.iloc[2:] = augmented_img_data

    augmented_rows.append(new_row)

augmented_df = pd.DataFrame(augmented_rows, columns=train.columns)

train = pd.concat([train, augmented_df], ignore_index=True)
print(f"증강 후 총 데이터 수: {len(train)}")
```
    고양이 이미지 9개, 토끼 이미지 12개를 증강합니다.
    고양이/토끼 이미지 증강: 100%|██████████| 21/21 [00:00<00:00, 3122.42it/s]증강 후 총 데이터 수: 790

확대 비율(zoom_factor)은 1.3배로 하고, 중앙 부분을 크롭하는 방식을 통해 증강을 했다. 이를 통해 토끼와 고양이를 헷갈려하던 모델에게 각각의 얼굴을 보고 학습을 더 할 수 있도록 했다.

```python
for lab in train['label']:
    tmp_df = train[train['label'] == lab]
    tmp_n = len(train[train['label']=='rabbit'])-len(tmp_df)
    if tmp_n > len(train[train['label']=='rabbit'])//2:
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
![Distributionn](/images/2025-03-11-image_classification/Distributionn.png)
이후 데이터의 개수가 가장 많은 label을 기준으로 개수를 맞춰 복사하는 방식의 업샘플링을 통해 데이터 불균형을 해결했다.

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
    100%|██████████| 1280/1280 [03:54<00:00,  5.46it/s]원본 데이터 크기: 1400, 증강 후 데이터 크기: 16800

데이터 불균형을 해소했으니, 증강을 할 차례이다. 평행 이동, 가우시안 노이즈 추가 이 두 가지의 방법을 사용했다. 회전은 제출 시 점수가 떨어지는 것을 보고 제거했더니 오히려 점수가 올라서 코드에서 아예 지웠고, 밝기 조정은 강도를 약하게 하면 의미가 있을 것이라는 생각이 아직 남아있어서 주석처리만 해놓았다. 증강률(n_augmented)은 다양하게 시도를 해보았지만 현재 리더보드상 가장 높은 점수는 증강률을 11로 했을 때가 0.96으로 좋았다.


## Train
```python
model = XGBClassifier(
    n_estimators=20000,
    learning_rate=0.08,
    max_depth=5,
    random_state=42,
    eval_metric='mlogloss',
    early_stopping_rounds=100,
    eta=0.01,
    tree_method='gpu_hist')

print("모델 학습 중...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=True)
```

학습 시 XGBoost 모델을 사용했다. 초기에는 LightGBM과 함께 하이퍼파라미터를 조정하며 테스트를 했는데, LightGBM은 데이터 증강률이 높을 수록 속도는 조금 더 빠른데 Validation Accuracy가 상대적으로 조금 낮게 나와서 XGBoost 를 채택하였다. \
약 10epoch 정도는 돌길 바라며 n_estimator=20000 로 설정했지만, 학습 로그를 보면 1668까지만 돌다가 멈춰버린다. 아마 early_stopping_rounds=100 파라미터 때문인 것 같은데, 이러면 1epoch 정도 밖에 돌지 않아 학습이 제대로 이루어진게 맞는지 의문이 들었다. 이 부분은 차차 공부해보기로 했다. \ 
다른 주요 하이퍼파라미터로는 모델의 학습률인 learning rate 는 0.08, 최대 트리의 깊이를 설정하는 max_depth 는 5, 트리의 견고함을 설정해주는 eta 는 0.01로 설정했다. eta는 값이 작을수록 모델이 견고해져서 오버피팅 방지에 유리하다. 


## Validation
```python
y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
print("\nValidation Accuracy: ", valid_accuracy, '%', sep='')
```
    /usr/local/lib/python3.11/dist-packages/xgboost/core.py:158: UserWarning: [07:29:13] WARNING: /workspace/src/common/error_msg.cc:27: The tree method `gpu_hist` is deprecated since 2.0.0. To use GPU training, set the `device` parameter to CUDA instead.

    E.g. tree_method = "hist", device = "cuda"

    warnings.warn(smsg, UserWarning)

    Validation Accuracy: 0.9517857142857142%

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
먼저 직접 데이터 증강을 해보고 성능 향상까지 이끌어낸 첫 경험이다 보니 데이터 전처리의 중요성을 더욱 깨닫게 되었다. 
수업에서 잠깐 배웠던 Data Augmentation 를 대회를 통해 직접 경험해보니 적은 데이터의 상황에서 필수적이라는 생각이 들었다.

#### 스스로에 대한 고찰
대회 초반에 또 제대로 된 EDA나 전처리 없이 모델링만으로 해서 첫 점수가 매우 낮았다. 
데이터를 분석하고 모델 성능 향상을 위해 전처리는 가장 중요한 작업임을 잊지 말아야 한다.

ㅡㅡㅡㅡ
3/11 기준 리더보드

![leaderb](/images/2025-03-11-image_classification/leaderb.png)
