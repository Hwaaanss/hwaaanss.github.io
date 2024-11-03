---
layout: single
title: "[데이터 분석] 2. 머신러닝 문제와 데이터 전처리"
excerpt: "데이터 분석에서 다루는 주요 머신러닝 문제와 데이터 전처리 과정"
mathjax: true
toc: true
toc_sticky: true
toc_label: "Basic of Data Analytics"
categories: Machine_Learning
tag: [DataScience, ML, Preprocessing]
---

## 1. 머신러닝 문제의 접근 방식(1)

### 1.1 시계열 예측(Forecasting)
시계열 예측은 시간의 흐름에 따라 기록된 데이터를 바탕으로 변수 간의 인과관계를 분석하여 미래를 예측하는 영역이다. 주요 활용 사례로는 날씨 예측, 주식 가격 예측, 상품 판매량 예측 등이 있다. 이 분야에서 사용되는 대표적인 알고리즘으로는 **ARIMA**와 **DeepAR**이 있다.

시계열 예측에서는 타겟 변수와 높은 상관관계를 가진 변수를 활용하여 예측 모델을 구축한다. 이 과정에서 **Regression**과 **Classification**이 응용될 수 있다.

### 1.2 추천 시스템(Recommendation System)
추천 시스템은 **Netflix Prize**를 통해 주목받았으며, 두 가지 주요 방식이 있다:
- **협업 필터링(Collaborative Filtering)**: 비슷한 취향의 사용자를 찾는 방식으로, 사용자와 취향이 비슷한 친구들이 본 영화를 찾는 것과 같다.
- **콘텐츠 기반 필터링(Content-Based Filtering)**: 사용자가 좋아할 만한 콘텐츠를 찾아내는 방식이다, 사용자가 좋아하는 장르, 감독, 키워드 등을 찾는 것과 같다.

추천 시스템에서는 희소성(Sparsity) 문제가 크며, 이를 해결하기 위해 **Matrix Factorization**과 **Factorization Machines** 등이 사용된다.

### 1.3 이상 탐지(Anomaly Detection)
이상 탐지는 공정 프로세스 관리, 금융 사기 거래 탐지 등에서 많이 사용된다. 정상 데이터를 기준으로 벗어나는 데이터를 **Abnormal**로 정의하며, 이를 탐지하기 위해 **Outlier Detection**, **Out-of-Distribution**, **One Class Classification** 등의 방법이 사용된다.

---

## 2. 머신러닝 문제의 접근 방식(2)

### 2.1 이미지 프로세싱(Image processing)
Classification - Localization - Object Detection - Instance Segmantation 등의 프로세스가 있다.

#### 산업에서 활용
제조업에서는 양/불량품 자동 판정 모델 이용
차량에서 카메라 센서를 이용해 사람이나 장애물 식별
OCR을 활용하여 아날로그에서 디지털로 전환에 편리성 제공(금융앱 신분증 촬영 등)

### 2.2 NLP(Natural Language Processing, 자연어 처리)
컴퓨터가 사람의 언어를 처리하는 모든 기술로 대표적으로 감정 분석(Sentiment analysis), STT(Speech To Text), 챗봇과 같은 것들이 있다.

#### 산업에서 활용
제품 리뷰에서 부정적인 평가 비율을 관리하여 상품의 평판 관리
다양한 미디어 매체에서 부정적인 의견을 모니터링하여 회사의 평판 관리
CS 업무 중 반복적인 질문, 복잡도가 높지 않은 질문들에 대한 자동 응대(챗봇) 

---

## 3. 데이터 전처리(Data Cleaning)

### 3.1 결측치 처리(Missing Value)
실제 데이터에서는 **N/A**나 **Null** 값이 빈번하게 발생한다. 이런 값들은 분석에 방해가 될 수 있기 때문에 적절한 처리가 필요하다. 예를 들어, Zero 비율이 0.95 이상인 경우 해당 컬럼을 분석에서 제외하거나 필요성을 재검토해야 한다.

### 3.2 다중공선성 문제(Multicollinearity)
독립 변수들 간의 상관관계가 높을 경우 다중공선성 문제가 발생할 수 있다. 이 문제를 해결하지 않으면 모델의 신뢰도와 예측력이 떨어지게 된다.

### 3.3 클래스 불균형(Class Imbalance)
대부분의 실제 데이터는 클래스 불균형을 가지고 있다. 클래스 비율이 0.9 이상인 경우에는 샘플링 방법을 고려해야 한다. 일반적으로 **Undersampling**과 **Oversampling**이 사용되며, 데이터 분포를 최대한 유지하면서 밸런스를 맞추는 것이 중요하다.

---

## 4. 특징 엔지니어링(Feature Engineering)

### 4.1 인코딩(Encoding)
컴퓨터는 정보의 개념을 이해하지 못하기 때문에 데이터를 숫자로 변환해야 한다. 카테고리 데이터는 **Label Encoding**이나 **One-hot Encoding**으로 변환할 수 있다.
- **Label Encoding**: 알파벳 순서대로 인덱스를 할당한다. "My name is Hwan." 의 경우 My:0, name:1, is:2, Hwan:3, .:4 가 된다.
- **One-hot Encoding**: 등장하는 데이터의 사전을 만들어 이진수와 유사하게 표현한다. "Hi everyone." 의 경우 Hi:[1,0,0], everyone:[0,1,0], .:[0,0,1] 이 된다. 

### 4.2 이산화(Binning)
연속형 변수는 왜도가 높거나 정규분포가 아닐 가능성이 높다. 이 경우 변수 값을 몇 개의 그룹으로 나누는 **Equal Width Binning**이나 **Equal Frequency Binning**을 사용하여 효율적인 분석이 가능하다.

**Equal Width Binning**은 데이터를 동일한 간격으로 나누는 것이다. 하지만 outlier에 의한 정상 데이터들이 한쪽으로 밀리는 영향을 많이 받는다. 또한 skewed data의 경우 한쪽에 몰린 데이터들이 한 카테고리로 몰리기 때문에 다룰 수 없다.
**Equal Frequency Binning**은 데이터를 각 구간에 속한 데이터의 개수가 동일하게 나누는 것이다.

### 4.3 스케일링(Scaling)
값의 범위가 큰 경우 계산 중 값이 발산할 수 있으므로, 이를 방지하기 위해 **MinMax Scaler**, **Standard Scaler** 등을 사용하여 스케일링을 한다.

**Standard Scaler**: 변수 각각을 $z_{i} \sim \mathcal{N}(0,1)$로 만드는 스케일링 기법이다. 사이킷런에서 이를 지원하는데, **from sklearn.preprocessing import StandardScaler**를 통해 불러올 수 있다.
**MinMax Scaler**: 변수 각각에 $\frac{x_{i}-\text{min}(x)}{\text{max}(x)-\text{min}(x)}$를 적용하는 스케일링 기법이다. 이 역시 사이킷런에서 지원하는데, **from sklearn.preprocessing import MinMaxScaler**를 통해 불러올 수 있다.

---

## 5. 특징 선택(Feature Selection)

### 5.1 상관 분석(Correlation Analysis)
상관 분석은 두 변수가 얼마나 밀접하게 관련되어 있는지를 평가하는 방법이다. **Pearson Correlation**과 **Spearman Correlation**이 대표적으로 사용된다.
- **Pearson Correlation**은 공분산을 표준화한 상관계수이다:
  $$
  p(X, Y) = \frac{cov(X, Y)}{\sqrt{var(X) \cdot var(Y)}}
  $$
  $$
  cov(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \mu_X)(Y_i - \mu_Y)
  $$

### 5.2 중요도 분석(Feature Importance)
**Feature Importance**는 모델에 가장 큰 영향을 주는 변수를 찾는 기법이다. **Permutation Importance**와 **SHAP**(Shapley Additive Explanation) 등이 사용된다.

---

## 6. 데이터 샘플링(Data Sampling)

### 6.1 Undersampling
타겟 데이터의 불균형을 줄이기 위해 사용되며, 다수 클래스의 데이터를 줄이는 방식이다. **Random Undersampling**, **Near Miss**, **Tomek Links** 등이 있다.

### 6.2 Oversampling
적은 클래스 데이터를 늘리는 방법이다. **SMOTE**(Synthetic Minority Oversampling Technique)는 KNN을 이용해 새로운 데이터를 생성하는 방법이다.

### 6.3 샘플링 결합 기법
**SMOTEENN**과 **SMOTETOMEK**는 각각 **SMOTE**와 **ENN**, **Tomek Links**를 결합하여 데이터를 조정하는 기법이다.

---

## 7. 학습 및 검증 데이터셋 분할(Train/Validation/Test)

### 7.1 데이터 분할 방법
전체 데이터를 **Train** 데이터와 **Test** 데이터로 나눈다. **Validation** 데이터셋을 추가하여 모델의 성능을 평가하고, **Overfitting**을 방지할 수 있다.

### 7.2 교차 검증(Cross Validation)
**K-fold Cross Validation**은 데이터셋을 K개의 폴드로 나눠 검증하는 기법이다. **StratifiedKFold**는 라벨의 분포를 고려하여 폴드를 나누는 방식이다.

---

## 8. 결론
데이터 분석의 기본 개념과 절차는 다양한 실제 데이터 문제에 적용될 수 있다. 머신러닝 모델의 성능을 높이기 위해서는 적절한 데이터 전처리와 특징 선택이 필수적이다. 데이터의 특성을 이해하고 적절히 처리하는 것이 성공적인 데이터 분석 프로젝트의 핵심이다.

#### 출처
패스트 캠퍼스 : 50개 프로젝트로 완벽하게 끝내는 머신러닝 시그니쳐 (https://fastcampus.co.kr/data_online_msignature)