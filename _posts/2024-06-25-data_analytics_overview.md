---
layout: single
title: "[데이터 분석] 1. AI/ML 시대 흐름과 실무 사례"
excerpt: "AI/ML 시대의 흐름과 데이터 분석 성공 및 실패 사례"
mathjax: true
toc: true
toc_sticky: true
toc_label: "Data Analytics Overview"
categories: Machine_Learning
tag: [DataScience, ML]
---

## 1. 데이터 분석의 시대 흐름

### 1.1 AI/ML의 발전 과정
AI/ML은 결코 하루아침에 등장한 기술이 아니다. 수세기에 걸쳐 축적된 수학적 이론을 기반으로 발전해 왔으며, 최근 100년 동안의 하드웨어와 소프트웨어의 발전이 이를 가능하게 했다. 초기에는 Turing Test, Nearest Neighbor와 같은 기초적인 개념이 등장하였고, 이후 Neural Network와 같은 고급 알고리즘이 추가되었다. 2010년대 후반에는 **BERT**와 **GPT** 같은 초대형 언어 모델이 도입되며 AI/ML의 영역이 확장되었다.

### 1.2 Software 1.0과 2.0의 패러다임 변화
AI의 발전과 함께 Software 1.0에서 Software 2.0으로의 전환이 이루어졌다. Software 1.0은 사람이 논리를 작성하고 코드를 구현하는 방식이었으나, **Software 2.0**은 데이터를 기반으로 기계가 스스로 학습하며 논리를 개발하는 방식이다. 이로 인해 테슬라 자율주행 시스템, 초대형 언어 모델과 같은 혁신적인 기술이 가능하게 되었다.

---

## 2. 실제 데이터 분석의 현실과 오해

### 2.1 현업 데이터 분석 프로세스
이상적인 데이터 분석 프로세스는 문제 정의, 데이터 확인, PoC, 그리고 정식 프로젝트로 이어진다. 하지만 실제 현업에서는 프로젝트 진행 중 지속적인 문제 재정의가 이루어지며, PoC 결과에 따라 정식 프로젝트로 채택되거나 종료될 수 있다.

### 2.2 데이터 분석가와 현업의 이해 차이
현업과 데이터 분석가 간에는 데이터 분석에 대한 이해의 차이로 인한 의사소통 문제와 괴리감이 자주 발생한다. 예를 들어, 데이터 분석가는 **EDA(Exploratory Data Analysis)**, 데이터 불균형, **Bias**, **Outlier** 등의 용어를 사용하지만, 현업 담당자들은 이를 충분히 이해하지 못할 수 있다. 이로 인해 프로젝트의 방향성이 일관되지 않게 되는 경우가 많다.

### 2.3 상관관계와 인과관계의 차이
데이터 분석 과정에서 상관관계와 인과관계는 명확히 구분되어야 한다. 예를 들어, 아이스크림 판매량과 기온 사이에는 상관관계가 높지만 인과관계는 아니다. 반면, 해변에 몰리는 인파는 상관관계와 인과관계 모두 크며, 통제가 가능한 변수이다.

---

## 3. 기업에서의 데이터 분석 실패 사례

### 3.1 기술 도구의 선택
데이터 분석에서 실패의 원인 중 하나는 적절하지 않은 도구의 선택이다. 강력한 도구가 항상 적합한 것은 아니며, 상황에 맞는 기술을 선별할 줄 아는 안목이 필요하다. '냉장고를 옮기기 위해 15톤 트럭이 필요하지 않다'는 비유가 이를 잘 설명한다. 모든 문제에 **Deep Learning**을 적용할 수 없으며, 문제에 적합한 기술이 사용되어야 한다.

### 3.2 엔지니어와 분석가의 역할 분담
현업에서 엔지니어와 데이터 분석가의 역할(R&R)이 명확히 구분되지 않는 경우가 많다. 분석가는 개발된 모델을 제공하지만, 시스템 운영자는 데이터 누락, 서버 이슈 등을 다뤄야 하며 이 과정에서 충돌이 발생할 수 있다.

### 3.3 데이터 중심 접근법의 중요성
과거에는 AI 시스템의 에러 해결에서 모델 최적화와 신규 알고리즘 개발이 주를 이루었다. 그러나 최근에는 **데이터 개선**이 더 중요하다는 인식이 확산되고 있다. Andrew Ng는 AI 프로젝트 성공의 80%는 데이터 정리와 관련되어 있으며, 나머지 20%는 모델이나 알고리즘과 관련이 있다고 언급했다.

---

## 4. 데이터 분석 성공 사례: A 전자 사례

### 4.1 프로젝트 개요
A 전자는 Rule-based 모델을 Deep Learning 모델로 전환하고자 하였다. 이 과정에서 높은 정확도뿐만 아니라 미탐지의 최소화와 빠른 추론 속도가 중요하게 고려되었다. PoC 단계에서 **VGG** 모델이 성능은 뛰어났지만, 추론 속도가 느려 **MobileNet V2**를 사용하여 개선하였다.

### 4.2 검증과 문제 해결
현장 테스트에서 모델의 학습 데이터 정확도는 95.72%였으나, 실제 공장에서는 50%로 떨어지는 문제가 발생하였다. 이 외에도 처리 시간 초과와 메모리 누수 현상이 발견되었다. 가장 중요한 문제는 라벨 불일치로 인한 미탐지였다. 이를 해결하기 위해 데이터 재분류 및 재모델링 작업이 수행되었다.

### 4.3 배포 및 최종 결론
프로젝트는 공장의 **FA(Factory Automation) 망**과 **OA(Office Automation) 망**이 분리된 환경에서 DMZ 구간을 통해 배포되었다. 최종적으로 PoC 후 6개월 만에 프로젝트가 성공적으로 완료되었다.

---

## 5. 결론 및 교훈

### 5.1 교훈
데이터 분석 프로젝트의 성공은 데이터의 준비와 문제 정의에 달려 있다. 분석가는 기술적 이해뿐만 아니라 프로젝트의 실무적 환경을 파악하고 이를 조율할 줄 알아야 한다. '모델링만 하는 분석가는 매력이 없다'는 말처럼 분석가는 데이터 엔지니어링 능력과 협업 능력을 함께 갖추어야 한다.

### 5.2 Data-Centric Approach의 중요성
모델의 성능이 기대에 미치지 못할 때는 알고리즘을 탓하기보다 데이터 자체를 점검하는 것이 우선이다. 이 접근법은 향후 데이터 분석 및 AI 프로젝트의 핵심 전략으로 자리 잡을 것이다.


#### 출처
패스트 캠퍼스 : 50개 프로젝트로 완벽하게 끝내는 머신러닝 시그니쳐 (https://fastcampus.co.kr/data_online_msignature)