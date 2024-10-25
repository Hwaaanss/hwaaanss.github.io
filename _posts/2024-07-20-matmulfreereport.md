---
layout: single
title: "Scalable MatMul-free Language Modeling Report Review by hwaaanss"
---


# Scalable MatMul-free Language Modeling 논문 분석

"Scalable MatMul-free Language Modeling" 논문은 대형 언어 모델(LLM)에서 자주 사용되는 행렬 곱셈(MatMul) 연산을 제거하면서도 고성능을 유지할 수 있는 새로운 모델 구조를 제안하고 검증하는 데 중점을 둔다. LLM의 계산량이 증가함에 따라 MatMul 연산은 LLM 전체 연산의 주요 비중을 차지하게 되며, 이는 모델의 파라미터 수와 맥락 길이가 증가할수록 더 큰 문제로 작용한다. 본 연구는 LLM에서 MatMul을 없애고도 기존 Transformer 기반 모델과 유사한 성능을 유지하는 ‘MatMul-free’ 모델을 구현해낸다. 이로 인해 필요한 메모리 자원을 절약하면서, GPU와 FPGA 등의 하드웨어에서 효율적인 모델 구현이 가능해졌다.

## 연구 배경 및 필요성

LLM에서는 대규모의 행렬 연산이 필수적이다. 일반적인 Transformer 모델은 각 토큰에 대해 Query, Key, Value (QKV) 행렬을 생성하여 이들 간의 곱셈을 통해 Attention 값을 구한다. 또한, 모델 파라미터 수가 수십억 단위로 증가하면 GPU나 TPU 등 하드웨어에서 고비용 연산과 높은 메모리 사용을 초래하게 된다. 이 논문에서는 MatMul-free LLM을 도입함으로써 이러한 문제를 해결하고자 한다.

### 기존 행렬 곱셈의 부담

기존의 LLM에서 MatMul 연산은 다차원 행렬과 벡터 연산을 수행하는데, 이 과정에서 대규모의 데이터 이동 및 동기화 작업이 발생한다. 예를 들어, Transformer 기반 모델의 Attention 연산은 $Q \times K^T$ 와 같은 행렬 곱셈이 필수적이며, 이는 모델의 각 계층(layer)에서 수십억 개 이상의 연산을 요구한다. 이에 따라 계산 비용이 매우 높아지며, 메모리 대역폭 제한과 과도한 전력 소모 등의 문제가 발생한다.

## MatMul-free 모델 구조 설계

본 연구는 MatMul-free 모델을 설계하기 위해 기본적인 Dense layer와 Attention 메커니즘을 대체하는 방법론을 제시한다. 이들은 다음과 같이 대체된다.

1. **Ternary Weights를 통한 Dense Layer 최적화**  
   Dense layer에서 자주 사용되는 MatMul 연산을 ternary (삼진) 가중치로 전환하여 {−1, 0, +1}의 값만을 사용하도록 설계하였다. 이렇게 함으로써 MatMul 연산은 단순한 덧셈과 뺄셈으로 변환될 수 있다. 일반적인 Dense layer에서 입력 \(x \in \mathbb{R}^{1 \times d}\)와 가중치 \(W \in \mathbb{R}^{d \times m}\) 간의 행렬 곱을 다음과 같이 표현할 수 있다:

   \[
   y = xW = \sum_{j=1}^{d} x_j W_{ij} \quad \text{for } i = 1, \dots, m
   \]

   여기서 \(W_{ij}\)를 {-1, 0, +1}로 제한하면 \(y = \sum x_j W_{ij}\) 연산이 단순한 덧셈으로 대체된다. 이로 인해 연산량이 크게 감소하고 메모리 사용량도 줄어든다.

2. **MatMul-free Self-Attention 메커니즘**  
   Self-Attention의 MatMul 연산을 Hadamard 곱으로 대체하여 토큰 간의 관계를 계산한다. 이를 위해 Gate Recurrent Unit (GRU)을 사용해 토큰 믹싱(token mixing)을 구현하고, 전체 Attention 연산을 element-wise 곱셈으로 처리하도록 한다. 이를 통해 기존의 Transformer와 유사한 수준의 정보 통합 효과를 유지하면서도 MatMul 연산을 완전히 제거하였다.

## 하드웨어 효율성 비교

논문에서는 제안한 MatMul-free 모델의 하드웨어 효율성을 GPU 및 FPGA 구현을 통해 검증하였다. 특히, GPU에서는 최적화된 CUDA 커널을 사용해, 메모리 사용량을 기존 모델에 비해 최대 61% 줄였고, 연산 속도도 4.57배 개선하였다. FPGA에서는 MatMul-free 모델을 위한 사용자 정의 하드웨어 아키텍처를 설계하여, 전체 전력 소모를 약 13W 수준으로 제한하고, 대형 언어 모델을 고효율로 처리할 수 있음을 보여주었다.

1. **GPU에서의 효율성**  
   실험에서 MatMul-free 모델은 기존의 Transformer++ 모델에 비해 연산량이 약 25.6% 감소하였으며, 대규모 모델일수록 메모리 절감 효과가 커졌다. GPU 메모리 소비는 최대 10배까지 감소하여, 기존에 메모리 부족으로 인해 배치 크기를 줄여야 했던 문제를 해결하였다.

2. **FPGA에서의 효율성**  
   FPGA 상에서는 Ternary 가중치를 효율적으로 처리하기 위한 전용 하드웨어 유닛을 개발하였다. 각 연산을 병렬적으로 처리할 수 있는 구조로, 단일 FPGA 코어에서는 전력 소모가 13.67W로 나타났으며, 고속 버스트 전송을 통해 최대 64배의 속도 개선을 기대할 수 있는 구조임을 입증하였다. 이를 통해 연산이 많은 대형 모델에서도 전력 소모가 적고 높은 처리량을 보일 수 있음을 확인하였다.

## 성능 평가 및 실험 결과

본 연구에서는 370M, 1.3B, 2.7B 파라미터 모델을 대상으로 성능을 평가하였다. 이를 통해 Transformer++ 모델과 비교하여 MatMul-free 모델의 확장법칙(scaling law)을 분석하였고, 결과는 다음과 같다.

1. **모델 크기에 따른 확장성**  
   모델의 파라미터가 증가함에 따라 MatMul-free 모델의 성능이 더욱 향상되었으며, 특히 2.7B 파라미터 모델에서는 Transformer++와 성능 격차가 거의 없어졌다. 이는 MatMul-free 구조가 대규모 파라미터에서도 충분히 유효함을 보여준다.
   
2. **다양한 벤치마크 테스트**  
   ARC-Easy, Hellaswag, Winogrande 등 다양한 언어 모델 벤치마크에서 기존 Transformer++와 유사한 성능을 기록하였고, ARC-Challenge와 OpenbookQA에서는 MatMul-free 모델이 오히려 우수한 결과를 보였다. 이는 특히 대규모 연산이 필요한 질문 응답 및 상식 추론 작업에서도 MatMul-free 모델이 높은 성능을 유지할 수 있음을 시사한다.

## 결론 및 기여

MatMul-free LLM은 LLM에서 자주 사용되는 MatMul 연산의 비효율성을 극복하고, 경량화된 연산을 통해 하드웨어 효율성을 극대화하는 새로운 접근 방식을 제안한다. 이 연구는 다음과 같은 기여를 제공한다.

- MatMul-free 구조가 고성능을 유지하면서도 메모리와 계산 자원을 절감하는 데 매우 효과적임을 입증하였다.
- 하드웨어 효율성이 높아 기존 GPU, FPGA뿐만 아니라 메모리 제약이 있는 시스템에서도 LLM을 효과적으로 실행할 수 있다.
- 향후 MatMul-free 모델의 확장 가능성을 입증함으로써, 차세대 경량 LLM을 위한 하드웨어 최적화의 방향성을 제시한다.

결과적으로 이 연구는 LLM이 꼭 고비용의 MatMul 연산에 의존할 필요가 없으며, 더 적은 연산으로도 효율적이고 성능 좋은 LLM을 설계할 수 있는 가능성을 제시한다.
```
