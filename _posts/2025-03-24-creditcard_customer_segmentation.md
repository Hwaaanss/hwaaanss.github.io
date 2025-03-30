---
layout: single
title: "신용카드 고객 세그먼트 분류 AI 경진대회(작성 중)"
excerpt: "신용카드 고객 분류 대회 참가 기록"
mathjax: true
toc: true
toc_sticky: true
toc_label: Code
categories: Dacon
tag: [AI, ML, DS, Competition]
---


## 대회 주제
일정 기간동안의 고객 신용정보, 승인매출, 청구, 잔액, 채널, 마케팅, 성과 등 다양한 데이터를 바탕으로 신용카드 고객 세그먼트(Segment) 분류 AI 알고리즘 개발

```python
import pandas as pd
import numpy as np
import gc
import matplotlib.pylab as plt
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')
```

## EDA(Exploratory Data Analysis)
#### Data Shape

고객별 금융활동 데이터 \
각 정보 내 parquet파일로 구성 \
1.회원정보[폴더] \
2.신용정보[폴더] \
3.승인매출정보[폴더] \
4.청구입금정보[폴더] \
5.잔액정보[폴더] \
6.채널정보[폴더] \
7.마케팅정보[폴더] \
8.성과정보[폴더] \
Target : 고객 세그먼트(A~E) \

로 구성되어 있으며, 각 파일들에는 "기준년월, ID, 남녀구분코드, 연령, 회원여부_이용가능, 회원여부_이용가능_CA, 회원여부_이용가능_카드론, 소지여부_신용, 소지카드수_유효_신용, 소지카드수_이용가능_신용, 입회일자_신용, 입회경과개월수_신용, 회원여부_연체" 등과 같은 컬럼 정보 858개가 있다. 이 각각의 컬럼 별로 행은 2,400,000개가 있다.  \
즉 모든 파일들을 이어 붙히면 (2400000,858) 의 shape인 셈이다.

## Data Preprocessing
 *전처리 과정의 baseline은 데이콘에서 제공된 코드를 바탕으로 작업하였음.

#### Data Load
```python
data_splits = ["train", "test"]

data_categories = {
    "회원정보": {"folder": "1.회원정보", "suffix": "회원정보", "var_prefix": "customer"},
    "신용정보": {"folder": "2.신용정보", "suffix": "신용정보", "var_prefix": "credit"},
    "승인매출정보": {"folder": "3.승인매출정보", "suffix": "승인매출정보", "var_prefix": "sales"},
    "청구정보": {"folder": "4.청구입금정보", "suffix": "청구정보", "var_prefix": "billing"},
    "잔액정보": {"folder": "5.잔액정보", "suffix": "잔액정보", "var_prefix": "balance"},
    "채널정보": {"folder": "6.채널정보", "suffix": "채널정보", "var_prefix": "channel"},
    "마케팅정보": {"folder": "7.마케팅정보", "suffix": "마케팅정보", "var_prefix": "marketing"},
    "성과정보": {"folder": "8.성과정보", "suffix": "성과정보", "var_prefix": "performance"}
}

months = ['07', '08', '09', '10', '11', '12']

for split in data_splits:
    for category, info in data_categories.items():
        folder = info["folder"]
        suffix = info["suffix"]
        var_prefix = info["var_prefix"]

        for month in months:
            file_path = f"drive/MyDrive/데이콘/신용카드고객세그먼트/{split}/{folder}/2018{month}_{split}_{suffix}.parquet"
            variable_name = f"{var_prefix}_{split}_{month}"
            globals()[variable_name] = pd.read_parquet(file_path)
            print(f"{variable_name} is loaded from {file_path}")
gc.collect()
```
데이터 분할(폴더) 구분하고, 각 데이터 유형별 폴더명, 파일 접미사, 변수 접두어 설정하였다. 각 파일들을 구글 드라이브에서 불러오는 작업이다. 

#### Data Merging
```python
info_categories = ["customer", "credit", "sales", "billing", "balance", "channel", "marketing", "performance"]
months = ['07', '08', '09', '10', '11', '12']

train_dfs = {}

for prefix in info_categories:
    df_list = [globals()[f"{prefix}_train_{month}"] for month in months]
    train_dfs[f"{prefix}_train_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_train_df is created with shape: {train_dfs[f'{prefix}_train_df'].shape}")


customer_train_df = train_dfs["customer_train_df"]
credit_train_df   = train_dfs["credit_train_df"]
sales_train_df    = train_dfs["sales_train_df"]
billing_train_df  = train_dfs["billing_train_df"]
balance_train_df  = train_dfs["balance_train_df"]
channel_train_df  = train_dfs["channel_train_df"]
marketing_train_df= train_dfs["marketing_train_df"]
performance_train_df = train_dfs["performance_train_df"]

gc.collect()
```

    [출력]
    customer_train_df is created with shape: (2400000, 78)
    credit_train_df is created with shape: (2400000, 42)
    sales_train_df is created with shape: (2400000, 406)
    billing_train_df is created with shape: (2400000, 46)
    balance_train_df is created with shape: (2400000, 82)
    channel_train_df is created with shape: (2400000, 105)
    marketing_train_df is created with shape: (2400000, 64)
    performance_train_df is created with shape: (2400000, 49)
    0

```python
train_df = customer_train_df.merge(credit_train_df, on=['기준년월', 'ID'], how='left')
print("Step1 저장 완료: train_step1, shape:", train_df.shape)
del customer_train_df, credit_train_df
gc.collect()

merge_list = [
    ("sales_train_df",    "Step2"),
    ("billing_train_df",  "Step3"),
    ("balance_train_df",  "Step4"),
    ("channel_train_df",  "Step5"),
    ("marketing_train_df","Step6"),
    ("performance_train_df", "최종")
]

for df_name, step in merge_list:
    train_df = train_df.merge(globals()[df_name], on=['기준년월', 'ID'], how='left')
    print(f"{step} 저장 완료: train_{step}, shape:", train_df.shape)
    del globals()[df_name]
    gc.collect()
```

    [출력]
    Step1 저장 완료: train_step1, shape: (2400000, 118)
    Step2 저장 완료: train_Step2, shape: (2400000, 522)
    Step3 저장 완료: train_Step3, shape: (2400000, 566)
    Step4 저장 완료: train_Step4, shape: (2400000, 646)
    Step5 저장 완료: train_Step5, shape: (2400000, 749)
    Step6 저장 완료: train_Step6, shape: (2400000, 811)
    최종 저장 완료: train_최종, shape: (2400000, 858)

불러온 train 데이터셋을 merge 하여 저장하고, merge를 위해 사용했던 변수들은 메모리 관리를 위해 삭제했다.

```python
test_dfs = {}

for prefix in info_categories:
    df_list = [globals()[f"{prefix}_test_{month}"] for month in months]
    test_dfs[f"{prefix}_test_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_test_df is created with shape: {test_dfs[f'{prefix}_test_df'].shape}")


customer_test_df = test_dfs["customer_test_df"]
credit_test_df   = test_dfs["credit_test_df"]
sales_test_df    = test_dfs["sales_test_df"]
billing_test_df  = test_dfs["billing_test_df"]
balance_test_df  = test_dfs["balance_test_df"]
channel_test_df  = test_dfs["channel_test_df"]
marketing_test_df= test_dfs["marketing_test_df"]
performance_test_df = test_dfs["performance_test_df"]

gc.collect()
```

    [출력]
    customer_test_df is created with shape: (600000, 77)
    credit_test_df is created with shape: (600000, 42)
    sales_test_df is created with shape: (600000, 406)
    billing_test_df is created with shape: (600000, 46)
    balance_test_df is created with shape: (600000, 82)
    channel_test_df is created with shape: (600000, 105)
    marketing_test_df is created with shape: (600000, 64)
    performance_test_df is created with shape: (600000, 49)
    0

```python
test_df = customer_test_df.merge(credit_test_df, on=['기준년월', 'ID'], how='left')
print("Step1 저장 완료: test_step1, shape:", test_df.shape)
del customer_test_df, credit_test_df
gc.collect()

merge_list = [
    ("sales_test_df",    "Step2"),
    ("billing_test_df",  "Step3"),
    ("balance_test_df",  "Step4"),
    ("channel_test_df",  "Step5"),
    ("marketing_test_df","Step6"),
    ("performance_test_df", "최종")
]

for df_name, step in merge_list:
    test_df = test_df.merge(globals()[df_name], on=['기준년월', 'ID'], how='left')
    print(f"{step} 저장 완료: test_{step}, shape:", test_df.shape)
    del globals()[df_name]
    gc.collect()
```

    [출력]
    Step1 저장 완료: test_step1, shape: (600000, 117)
    Step2 저장 완료: test_Step2, shape: (600000, 521)
    Step3 저장 완료: test_Step3, shape: (600000, 565)
    Step4 저장 완료: test_Step4, shape: (600000, 645)
    Step5 저장 완료: test_Step5, shape: (600000, 748)
    Step6 저장 완료: test_Step6, shape: (600000, 810)
    최종 저장 완료: test_최종, shape: (600000, 857)

이어서 test 데이터셋도 merge 작업을 해준다. 컬럼의 수가 하나 적은 이유는 segment 컬럼이 빠졌기 때문이다.

#### Data Encoding
```python
feature_cols = [col for col in train_df.columns if col not in ["ID", "Segment"]]

X = train_df[feature_cols].copy()
y = train_df["Segment"].copy()

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

X_test = test_df.copy()

encoders = {}

for col in categorical_features:
    le_train = LabelEncoder()
    X[col] = le_train.fit_transform(X[col])
    encoders[col] = le_train
    unseen_labels_val = set(X_test[col]) - set(le_train.classes_)
    if unseen_labels_val:
        le_train.classes_ = np.append(le_train.classes_, list(unseen_labels_val))
    X_test[col] = le_train.transform(X_test[col])
gc.collect()

X, X_val, y_encoded, y_encoded_val = train_test_split(X, y_encoded, test_size=0.1, shuffle=True, stratify=y_encoded, random_state=42)
```
이제 병합한 train 데이터와 test 데이터들을 인코딩 했다.  \
먼저 train 데이터에서 ID와 Segment를 제거해서 X에 copy하고, target인 Segment는 y에 copy한 후, y는 LabelEncoder로 인코딩해줬다. \
그 후 test 데이터도 copy하고, LabelEncoder로 정수로 변환했다. 변환된 LabelEncoder 객체를 딕셔너리 형태로 저장한 후, train 데이터에 없던 새로운 범주값이 나오면 딕셔너리에 추가하도록 했다.  \
이후 학습 단계에서 validation을 위한 데이터를 train_test_split을 사용해서 나눴다. 마지막으로 병합 할 때와 마찬가지로 메모리 관리를 위해 gc.collect()로 사용했던 변수들을 제거했다.

#### Check Target's Distribution Before Train
```python
y.value_counts()
```

    [출력]
        count
    Segment	
    E	1922052
    D	349242
    C	127590
    A	972
    B	144

병합, train set과 validation set의 분리, 각 데이터셋의 인코딩 작업까지 마쳤으니 현재 데이터의 분포를 알아봐야겠다.\
코드를 통해 확인해보니 데이터 불균형이 광장히 심했다. 데이터 불균형을 맞추기 위해 오버샘플링을 하자니 코랩이나 로컬 환경의 RAM이 부족해서 안되고, 언더샘플링을 하자니 손실되는 데이터의 개수가 너무 많을 것 같아 결국 XGBoost 모델의 하이퍼파라미터들을 이용해서 해결해보고자 했다.

## Train
#### Train a New Model
```python
model = xgb.XGBClassifier(
    random_state=42,
    eta=0.1,
    n_estimators=1400,
    max_depth=6,
    reg_alpha=2,
    reg_lambda=8,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric='mlogloss'
)

model.fit(X, y_encoded, eval_set=[(X, y_encoded),(X_val, y_encoded_val)], verbose=1)
model.save_model('drive/MyDrive/데이콘/신용카드고객세그먼트/model_dump/xgboost_submit_0.1_1400_6_2_8_0.8_0.8_100_20.json')
```
각 하이퍼파라미터들에 대해 설명을 간단히 하겠다. \
eta(≒learning rate) : 일반적으로 아는 learning rate 라고 생각하면 된다. \
n_estimator : 트리 모델의 개수이다. epoch 혹은 step 처럼 학습의 횟수를 설정하듯이 설정하기도 한다. \
max_depth : 트리 모델의 최대 깊이이다. 너무 깊어지면 과적합이 일어날 수 있다. \
reg_alpha : L1-규제 즉, Lasso 파라미터이다. 가중치를 강제로 0으로 만들어버리기 때문에 feature가 많은 경우 feature selection 의 효과를 볼 수도 있다. 불필요한 데이터 feature들을 제거하도록 사용했다. \
reg_lambda : L2-규제 즉, Ridge 파라미터이다. 모든 특성들의 가중치를 줄여 오버피팅 방지에 사용된다. 나는 과적합 방지 장치를 걸어두고 epoch을 늘리는 방식을 택해서 설정해줬다.  \
subsample : 각 스탭마다 사용할 샘플의 비율이다. 오버피팅 방지를 위해 사용했다. \
colsample_bytree : 각 스탭마다 사용할 feature의 비율이다. 역시 과적합 방지를 위해 사용된다. \
early_stopping_rounds : 역시 과적합 방지를 위한 early stop 파라미터이다. 여태 학습 돌리면서 이거에 걸려서 중단된 적은 없었지만, 혹시 몰라서 설정했다. \
eval_metric : 이 모델은 다중 분류를 위한 모델이므로 mlogloss를 사용했다.  \
이 외로 scale_pos_weight 라는 클래스 불균형을 어느정도 해결해주는 하이퍼파라미터가 있어서 사용하고 싶었지만 이진 분류에만 사용이 가능해서 아쉬웠다. \
model.fit() 을 통해 학습을 진행한 후, 모델을 저장하도록 했다. 평소 저장은 마지막쯤에 답지와 같이 해왔지만 워낙 코랩 리소스를 많이 먹기 때문에 불안해서 저장부터 하도록 했다.

#### Train with a Pretrained Model
```python
model = xgb.XGBClassifier(
    random_state=42,
    eta=0.1,
    n_estimators=600,
    max_depth=6,
    reg_alpha=2,
    reg_lambda=8,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric='mlogloss'
)
model.load_model('drive/MyDrive/데이콘/신용카드고객세그먼트/model_dump/xgboost_submit_0.1_1400_6_2_8_0.8_0.8_20.json')
model.fit(X, y_encoded, eval_set=[(X, y_encoded),(X_val, y_encoded_val)], xgb_model=model, verbose=1)
model.save_model('drive/MyDrive/데이콘/신용카드고객세그먼트/model_dump/xgboost_submit_0.1_2000_6_2_8_0.8_0.8_100_20.json')
```
구글 코랩 TPU v28의 런타임이 학습을 한번 돌리면 아슬아슬하게 끝나기 때문에, 이어서 학습을 진행하기 위해 불러와서 돌리는 코드도 구성했다.

#### Check the Eval Metric Score
```python
results = model.evals_result()

plt.figure(figsize=(8, 5))

for dataset in results["validation_0"]:
    plt.plot(results["validation_0"][dataset], label=f"Train {dataset}")
for dataset in results["validation_1"]:
    plt.plot(results["validation_1"][dataset], label=f"Validation {dataset}")

plt.xlabel("Iteration")
plt.ylabel("Metric Value")
plt.title("XGBoost Eval Metric")
plt.legend()
plt.grid()
plt.show()
```

학습의 전체적인 과정과 문제가 있진 않았는지 확인을 위해 시각화를 했다. \
![metric_graph](/images/2025-03-24-creditcard_segmentation/metric_graph.png)   \
다행히 문제가 있어보이진 않는다.

## Prediction
```python
X_test.drop(columns=['ID'],inplace=True)
y_test_pred = model.predict(X_test)
y_test_pred_labels = le_target.inverse_transform(y_test_pred)

test_data = test_df.copy() 
test_data["pred_label"] = y_test_pred_labels
```
test 데이터에서 ID 컬럼을 제거하고, 학습시킨 모델을 통해 예측한 target들을 y_test_pred에 저장한 후, 인코딩 과정을 inverse 해서 답으로 되돌린다.(숫자->영문자) \
이후 원본 테스트 데이터프레임을 복사해 답지 부분을 만들어 넣는다.

## Submission
```python
submission = test_data.groupby("ID")["pred_label"] \
    .agg(lambda x: x.value_counts().idxmax()) \
    .reset_index()

submission.columns = ["ID", "Segment"]
submission.to_csv('drive/MyDrive/데이콘/신용카드고객세그먼트/submissions/xgboost_submit_0.1_1400_6_2_8_0.8_0.8_100_20.csv',index=False)
```
제출 파일을 만들고 저장하도록 하면서 코드는 끝이다.


## 돌아보기
#### 알게 된 점
(대회 종료 or 도전 종료 후 작성)

#### 스스로에 대한 고찰
(대회 종료 or 도전 종료 후 작성)

ㅡㅡㅡㅡ  \
3/24 기준 리더보드

![leaderbo0324](/images/2025-03-24-creditcard_segmentation/leaderbo0324.png)
2025.03.24 기준 1등 기록했다. reg_lambda 값을 올려 L2 규제를 강화했더니 점수가 더 올랐다.