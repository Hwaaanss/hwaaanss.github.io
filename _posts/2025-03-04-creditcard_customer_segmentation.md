---
layout: single
title: "신용카드 고객 세그먼트 분류 AI 경진대회"
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

고객별 금융활동 데이터
각 정보 내 parquet파일로 구성
1.회원정보[폴더]
2.신용정보[폴더]
3.승인매출정보[폴더]
4.청구입금정보[폴더]
5.잔액정보[폴더]
6.채널정보[폴더]
7.마케팅정보[폴더]
8.성과정보[폴더]
Target : 고객 세그먼트(A~E)

로 구성되어 있으며, 각 파일들에는 "기준년월, ID, 남녀구분코드, 연령, 회원여부_이용가능, 회원여부_이용가능_CA, 회원여부_이용가능_카드론, 소지여부_신용, 소지카드수_유효_신용, 소지카드수_이용가능_신용, 입회일자_신용, 입회경과개월수_신용, 회원여부_연체" 등과 같은 컬럼 정보 858개가 있다. 이 각각의 컬럼 별로 행은 2,400,000개가 있다. 즉 모든 파일들을 이어 붙히면 (2400000,858) 의 shape인 셈이다.

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
```
```python
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
    Step1 저장 완료: test_step1, shape: (600000, 117)
    Step2 저장 완료: test_Step2, shape: (600000, 521)
    Step3 저장 완료: test_Step3, shape: (600000, 565)
    Step4 저장 완료: test_Step4, shape: (600000, 645)
    Step5 저장 완료: test_Step5, shape: (600000, 748)
    Step6 저장 완료: test_Step6, shape: (600000, 810)
    최종 저장 완료: test_최종, shape: (600000, 857)

이어서 test 데이터셋도 merge 작업을 해준다. 컬럼의 수가 하나 적은 이유는 segment 컬럼이 빠졌기 때문이다.
