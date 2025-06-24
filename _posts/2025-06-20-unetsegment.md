---
layout: single
title: "U-Net 논문 리뷰"
excerpt: "U-Net : Convolutional Networks for Biomedical Image Segmentation"
mathjax: true
toc: true
toc_sticky: true
toc_label: Code
categories: Paper_Review
tag: [AI, DL, CV]
---

## Introduction
Semantic Segmentation이란 이미지 객체 분할에 있어서 의미(semantic)가 같은 객체라면 하나의 클래스로 분할(segmentation)하는 것을 의미한다.
![whatisunet](/images/2025-06-20-unet/whatisunet.png)
출처: DeeplabV3+ paper

이와 같이 같은 사람으로 의미가 같다면 사람이라는 하나의 클래스로 분류를 한다.

## Network Architecture
U-Net 모델의 아키텍처에 대해 알아보자.
![whatisunet](/images/2025-06-20-unet/unetarchitecture.png)
U-Net은 위 그림과 같이 전체적인 구조가 U자 모양을 하고 있어 U-Net이라는 이름이 붙었다. 구조는 크게 이미지를 압축하며 feature를 추출하는 **Encoder(Contracting Path)** 부분과, 다시 이미지를 확장하며 정확한 위치를 찾는 **Decoder(Expansive Path)** 부분으로 나눌 수 있다.

![Encoder and Decoder](/images/2025-06-20-unet/unet-encoder-decoder.png)
출처: U-Net Paper

### U-Net Encoder (Contracting Path)
인코더는 입력 이미지의 Context, 즉 전체적인 특징을 잡아내는 부분이다. 일반적인 CNN 모델의 특징 추출부와 유사한 역할을 한다고 볼 수 있다.

![U-Net Encoder Diagram](/images/2025-06-20-unet/unet-encoder-detail.png)
출처: Deep Campus tistory

U-Net의 인코더는 다음과 같은 과정으로 구성된다.
* **ConvBlock**: 3x3 Convolution 연산을 두 번 반복하고, 각 Conv 연산 후에는 Batch Normalization과 ReLU 활성화 함수를 적용한다. 이 과정을 통해 feature map을 추출하는 것이다.
* **EncoderBlock**: 위에서 설명한 ConvBlock을 거친 후, 2x2 Max Pooling을 통해 feature map의 가로, 세로 크기를 절반으로 줄인다. 이 과정에서 채널(filter)의 수는 2배로 늘어난다 (e.g., 64 -> 128).

이러한 EncoderBlock을 여러 번 반복하면서 이미지의 공간 정보는 압축되고, Context 정보는 더욱 풍부해진다.

### U-Net Decoder (Expansive Path)
디코더는 인코더에서 압축된 feature map을 다시 확장하여 원본 이미지 크기의 세분화된 분할 맵(Segmentation map)을 만드는 부분이다. 이 부분이 U-Net의 핵심 중 하나이다.

![U-Net Decoder Diagram](/images/2025-06-20-unet/unet-decoder-detail.png)
출처: Deep Campus tistory

디코더의 과정은 다음과 같다.
* **DecoderBlock**:
    1.  **ConvTranspose (Up-convolution)**: 2x2 Up-convolution을 통해 feature map의 가로, 세로 크기를 2배로 늘리고 채널 수는 절반으로 줄인다.
    2.  **Concatenate**: 여기서 U-Net의 가장 중요한 특징인 **Skip Connection**이 사용된다. 현재 단계의 feature map을 동일한 크기를 가졌던 Encoder단의 feature map과 채널 방향으로 결합(Concatenate)한다. 이를 통해 Encoder가 압축 과정에서 잃어버릴 수 있는 위치 정보(Localization)를 보완해 주는 것이다.
    3.  **ConvBlock**: 마지막으로 3x3 Convolution을 두 번 반복하여 feature를 정제한다.

이 과정을 반복하여 최종적으로 원본 이미지와 비슷한 크기의 Segmentation Map을 출력하게 된다.

### 3D U-Net
U-Net은 2D 이미지뿐만 아니라 3D 의료 영상(e.g., MRI, CT)에도 적용될 수 있도록 3D U-Net으로 확장되었다. 기본적인 U자 구조와 Skip Connection 개념은 동일하며, 모든 연산이 3D(Convolution, Max Pooling 등)로 확장된 형태이다.

![3D U-Net Architecture](/images/2025-06-20-unet/3d-unet.jpg)
출처: ResearchGate Unet Image by Rogger Booto Tokime

## Training
U-Net 논문에서 사용된 학습 방법과 전략에 대해 알아본다.

### Setting
논문에서 제안한 초기 학습 환경 설정은 다음과 같다.
* [cite_start]**Optimizer**: SGD (Stochastic Gradient Descent) 
* [cite_start]**Momentum**: 0.99 
* [cite_start]**Batch size**: 1 
* [cite_start]**Learning rate**: 0.01 
* [cite_start]**Weight Initialization**: He Initialization 

### Strategy for Enhance
적은 양의 데이터셋으로도 높은 성능을 내기 위해 U-Net은 몇 가지 독창적인 전략을 사용했다.

#### Weight map
의료 이미지는 종종 배경이 대부분을 차지하고 분할하려는 세포(객체)는 작은 영역을 차지하는 클래스 불균형 문제가 있다. 또한, 같은 종류의 세포들이 서로 붙어 있는 경우 경계를 구분하기 어렵다. [cite_start]이를 해결하기 위해 Loss에 가중치를 주는 Weight map을 사용했다.

[cite_start]$w(\mathbf{x}) = w_c(\mathbf{x}) + w_0 \cdot \exp\left(-\frac{(d_1(\mathbf{x}) + d_2(\mathbf{x}))^2}{2\sigma^2}\right)$ 

* [cite_start]$w_c(\mathbf{x})$ : 클래스 빈도의 불균형을 보정하기 위한 가중치이다.
* [cite_start]$w_0 \cdot \exp(...)$ : 서로 다른 객체가 맞닿아 있는 경계 부분에 높은 가중치를 부여하여 모델이 경계를 잘 학습하도록 유도한다. [cite_start]$d_1(\mathbf{x})$은 가장 가까운 객체의 경계까지의 거리, $d_2(\mathbf{x})$는 두 번째로 가까운 객체의 경계까지의 거리를 의미한다.

#### Data Augmentation : Elastic Deformation
적은 데이터로도 모델이 다양한 상황에 대응할 수 있도록 데이터 증강 기법을 사용했다. [cite_start]특히 의료 이미지는 조직의 변형이 많기 때문에, 이미지에 임의의 탄성 변형(Elastic Deformation)을 주어 모델의 강건함(Robustness)을 높였다.

![Elastic Deformation Example](/images/2025-06-20-unet/elastic-deformation.png)
[cite_start]출처: Elastic Deformations for Data Augmentation in Breast Cancer Mass Detection 

#### Data Extrapolate : Mirroring
U-Net은 이미지를 타일 단위로 처리하는데, 이 때 이미지의 경계 부분은 주변 픽셀 정보가 부족하여 예측이 불안정할 수 있다. [cite_start]이를 해결하기 위해 원본 이미지의 경계를 기준으로 주변을 거울처럼 반사(Mirroring)하여 패딩하는 방식을 사용했다. 이를 통해 경계 부분의 예측 정확도를 높였다.

![Mirroring Example](/images/2025-06-20-unet/mirroring.png)
[cite_start]출처: Code Journey github blog 

## Experiments
U-Net은 여러 이미지 분할 챌린지에서 뛰어난 성능을 입증했다.

### 1st Experiment : EM segmentation challenge
전자 현미경(EM) 이미지에서 뉴런 구조를 분할하는 챌린지이다. [cite_start]U-Net은 이 챌린지에서 다른 알고리즘들을 제치고 1위를 차지했다. [cite_start]특히 Warping Error에서 가장 낮은 수치를 기록했다.

![EM Challenge Results](/images/2025-06-20-unet/em-challenge-results.png)
[cite_start]출처: U-Net Paper 

### 2nd Experiment : ISBI cell tracking challenge
[cite_start]세포 분할 및 추적 챌린지에서도 U-Net은 뛰어난 성능을 보였다. [cite_start]PhC-U373 데이터셋과 DIC-HeLa 데이터셋 모두에서 월등히 높은 IoU 점수(각각 0.9203, 0.7756)를 기록하며 1위를 차지했다.

![ISBI Challenge Results](/images/2025-06-20-unet/isbi-challenge-results.png)
[cite_start]출처: U-Net Paper 

## Authors’ Conclusion & My Opinion
* [cite_start]**Authors' Conclusion**: 저자들은 "U-Net 아키텍처는 매우 다른 종류의 생물의학적 분할 응용 분야에서 아주 좋은 성능을 달성했다"고 결론 내렸다.
* **My Opinion**:
    * U-Net은 Skip Connection을 통해 인코더의 풍부한 지역 정보를 디코더로 전달하는 아이디어가 매우 훌륭하다고 생각한다. 이 구조는 이후 Segmentation 모델들의 표준처럼 자리 잡았다.
    * [cite_start]다만, Encoder의 feature map을 계속 유지해야 해서 메모리(RAM) 요구량이 크다는 단점이 있다.
    * [cite_start]최근에는 DALL-E, Midjourney, Stable Diffusion과 같은 이미지 생성 AI의 Diffusion model 내부에서 노이즈를 예측하는 부분에 U-Net 구조가 핵심적으로 사용되고 있다. Segmentation을 넘어 생성 모델의 근간이 되었다는 점에서 매우 중요한 모델이라고 생각한다.

## Implement Network
논문 리뷰를 바탕으로 직접 Brain Tumor MRI 데이터셋을 이용해 U-Net을 구현하고 하이퍼파라미터 튜닝을 진행해 보았다.

### Dataset : Brain Tumor MRI Images
[cite_start]뇌종양 MRI 이미지와 해당 영역을 표시하는 마스크(Mask)로 구성된 데이터셋을 사용했다.
* [cite_start]학습 데이터: 1502장 
* [cite_start]검증 데이터: 429장 
* [cite_start]테스트 데이터: 215장 

[cite_start]데이터가 충분하고, 종양의 위치 자체가 중요한 정보라고 판단하여 Elastic Deformation과 Mirroring은 적용하지 않았다.

![Brain Tumor Dataset Example](/images/2025-06-20-unet/brain-tumor-dataset.png)

### Setting
* [cite_start]**Optimizer**: SGD, AdamW 
* [cite_start]**Batch size**: 1, 4 
* [cite_start]**Learning rate**: 1e-2, 1e-4 
* [cite_start]**Epochs**: 10, 30, 50 
* [cite_start]**Loss function**: Dice Loss ($Dice = \frac{2 * |A \cap B|}{|A| + |B|}$) 
* [cite_start]**Scoring**: IoU Score ($IoU = \frac{A \cap B}{A \cup B}$) 

### Combinations of SGD Hyperparameters
![SGD settings](/images/2025-06-20-unet/table-sgd.png)
![SGD graph](/images/2025-06-20-unet/graph-sgd.png)

### Hyperparameter Tuning of SGD
[cite_start]SGD 옵티마이저를 사용했을 때, 가장 좋은 성능을 보인 조합은 **learning rate=1e-4, batch size=4, epoch=50** 이었고, 테스트 데이터셋에서 약 **0.544**의 Mean IoU 점수를 얻었다.
![SGD pred](/images/2025-06-20-unet/pred-sgd.png)

### Combinations of AdamW Hyperparameters
![AdamW settings](/images/2025-06-20-unet/table-adamw.png)
![AdamW graph](/images/2025-06-20-unet/graph-adamw.png)

### Hyperparameter Tuning of AdamW
[cite_start]AdamW 옵티마이저를 사용했을 때, 가장 좋은 성능을 보인 조합 역시 **learning rate=1e-4, batch size=4, epoch=50** 이었고, 테스트 데이터셋에서 약 **0.542**의 Mean IoU 점수를 얻었다.
![AdamW pred](/images/2025-06-20-unet/pred-adamw.png)

### Best Model
두 옵티마이저의 결과가 비슷했지만, 미세하게 더 높았던 SGD 모델을 최종 모델로 선정했다.
* [cite_start]**Optimizer**: SGD 
* [cite_start]**Learning rate**: 1e-4 
* [cite_start]**Batch size**: 4 
* [cite_start]**Epoch**: 50 
* [cite_start]**Test IoU Score**: **0.5446** 

![Final Best Model Graph](/images/2025-06-20-unet/best-sgd.png)
![Final Best Model Result](/images/2025-06-20-unet/pred-sgd.png)

## Resources
* [cite_start]U-Net Paper: https://arxiv.org/abs/1505.04597 
* [cite_start]U-Net encoder, decoder: https://pasus.tistory.com/204 
* [cite_start]3D U-Net: https://www.researchgate.net/figure/sual-representation-of-the-U-Net-network_fig1_337011086 
* [cite_start]Elastic deformation: https://web.fe.up.pt/~jsc/publications/conferences/2018EMecaBHI.pdf 
* [cite_start]Mirroring: https://joungheekim.github.io/2020/09/28/paper-review/ 
* [cite_start]Example of Semantic Segmentation: https://arxiv.org/abs/1802.02611 

---
## In closing
U-Net에 대한 논문 리뷰와 간단한 구현을 진행해 보았다. 이상으로 글을 마친다.