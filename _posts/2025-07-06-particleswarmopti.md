---
layout: single
title: "Particle Swarm Optimization (PSO) 탐구 기록 (작성중)"
excerpt: "하나가 아닌 여럿이서 최적의 답을 찾아나가는 최적화 알고리즘"
mathjax: true
toc: true
toc_sticky: true
toc_label: Contents
categories: Research
tags: [DL, Optimization]
---

## Idea
학교에서 들은 딥러닝 관련 강의에서 배웠던 딥러닝 관련 optimizer들은 SGD, RMSProp, AdaGrad, Adam이 있다. 하지만 "이 최적화 알고리즘들은 혼자서 움직이는데 local minima 에서의 탈출이 과연 실제 데이터풀에서도 이론처럼 잘 될까?" 라는 의문이 계속 들었다. 또한 단순한 최적화 기법으로만 알고 있다가 유튜버 혁펜하임의 영상을 보다가 최적화이론이라는게 있다는 것을 알게 되었다. 그래서 알아보니 내가 알고 있던 최적화 기법들은 현재는 정말 일반적인 기법이고, 더 고도화되거나 창의적인 아이디어를 이용한 기법들이 많은 것을 알고 호기심이 생겨 계속해서 공부를 이어갔다. 이 호기심은 자연에서 영감을 얻은 Metaheuristic 최적화 알고리즘으로 이어졌다. 그중에서도 새나 물고기 떼의 사회적 행동을 모방한 **Particle Swarm Optimization (PSO)**가 직관적이면서도 강력해 보여서, 이번 기회에 제대로 파헤쳐 보기로 마음먹었다.

## Details
**연구 정의**: 개별 입자들이 각자 탐색한 최적의 경험=과 집단 전체가 공유하는 최적의 경험=을 바탕으로, 문제 공간을 효율적으로 탐색하여 전역 최적해(Global Optimum)를 찾는 알고리즘을 이해하고 구현하는 것을 목표로 함

#### 알고리즘의 주요 학습 포인트
1.  **입자(Particle)의 상태**: 각 입자는 위치와 속도라는 두 가지 핵심 정보를 가진다. 위치는 문제에 대한 하나의 해답 후보이고, 속도는 이 후보가 다음 스텝에서 어느 방향으로 얼마나 이동할지를 결정하는 '벡터'라고 볼 수 있다
2.  **두 가지 핵심 정보**:
    * `pbest` (Personal Best): 한 입자가 탐색을 시작한 이래로 발견한 가장 좋았던, 즉 목적 함수 값이 가장 낮은 위치이다. 각 입자의 개인적인 기억으로도 볼 수 있다.
    * `gbest` (Global Best): 전체 입자들 중에서 가장 좋았던 위치이다. 집단 전체가 공유하는 최적의 의견(?) 이라고 볼 수 있다.
3.  **탐색 메커니즘**: 각 입자는 현재 자신의 속도를 유지하려는 관성, 자신이 찾았던 최고 지점(`pbest`)으로 돌아가려는 인지적 힘, 그리고 집단 전체가 인정한 최고 지점(`gbest`)으로 향하려는 사회적 힘 사이에서 다음 행보를 결정한다.

**알고리즘의 최종 목표**: 복잡한 다차원 함수의 최솟값을 찾는 과정을 시각화하고, PSO가 어떻게 최적해에 수렴하는지 직관적으로 이해.

## PSO Algorithm & Equations
PSO의 핵심 아이디어는 아래 두 개의 간단한 수식으로 요약된다. 각 입자($i$)는 매 타임스텝($t$)마다 이 수식에 따라 자신의 속도와 위치를 업데이트한다.

#### 1. 속도 업데이트 (Velocity Update)
입자의 다음 속도($v_{i}(t+1)$)는 세 가지 힘의 합으로 결정된다. 마치 여러 방향에서 잡아당기는 힘의 균형을 맞추는 것과 같다.

$$
v_{i}(t+1) = w \cdot v_{i}(t) + c_1 \cdot r_1 \cdot (\text{pbest}_{i} - x_{i}(t)) + c_2 \cdot r_2 \cdot (\text{gbest} - x_{i}(t))
$$

-   $w \cdot v_{i}(t)$: **관성(Inertia)**. "가던 길을 계속 가려는 성질"이다. 이 값이 크면 넓은 영역을 탐색하려는 경향이 강해지고, 작으면 현재 위치 주변을 세밀하게 탐색하려는 경향이 강해진다.
-   $c_1 \cdot r_1 \cdot (\text{pbest}_{i} - x_{i}(t))$: **인지적 요소(Cognitive Component)**. "그래도 내 경험상 거기가 제일 좋았지"라며 자신의 최고 경험으로 회귀하려는 힘으로 볼 수 있다. 개인의 고집이나 확신에 비유할 수 있다.
-   $c_2 \cdot r_2 \cdot (\text{gbest} - x_{i}(t))$: **사회적 요소(Social Component)**. "다들 저기가 제일 좋다더라"라며 집단의 성공을 따르려는 힘으로 비유할 수 있다. 대세를 따르거나 동조하는 사회적 행동과 닮았다.

여기서 $r_1, r_2$라는 랜덤 값이 곱해지는 점이 재미있다. 이 값들이 없다면 모든 입자는 `pbest`와 `gbest`를 향해 직선적으로만 움직일 것이다. 이는 너무 단조로워서 local optimum에 빠지기 쉽다. 랜덤 값을 통해 약간의 창의성 혹은 변덕을 부여함으로써, 입자들이 더 다양하고 예측 불가능한 경로로 탐색하게 하여 더 나은 해를 찾을 가능성을 열어주는 것이다.

#### 2. 위치 업데이트 (Position Update)
속도가 결정되면, 다음 위치($x_{i}(t+1)$)는 현재 위치에 새로운 속도를 더하여 간단하게 계산된다.

$$
x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)
$$

이 두 업데이트 과정을 모든 입자에 대해 반복하다 보면, 입자들은 점차 최적해 근처로 모여들게 된다. 개인적인 탐험과 사회적인 정보 공유 사이의 절묘한 균형이 집단 지성을 만들어내는 과정이 흥미를 돋군다.

## Pipeline & Implementation
코드로 직접 구현해보며 PSO의 작동 원리를 확인해 보았다. 2차원 함수 $f(x, y) = x^2 + y^2$ 의 최솟값(0, 0)을 찾는 예제를 구성했다.

#### Phase 1: PSO 알고리즘 구현
먼저 필요한 라이브러리를 임포트하고, PSO 알고리즘의 핵심 로직을 클래스로 구현했다.

1. 라이브러리 임포트

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
```

2. PSO 최적화기 클래스 정의 및 목적 함수 정의
알고리즘의 세부 파라미터와 로직을 담은 클래스를 설계하고, 최적화할 대상 함수를 정의했다. 목적 함수는 Rastrigin Function를 사용했다. Rastrigin Function는 최적화 알고리즘 벤치마크에 자주 사용되는 함수로, Global optimum은 하나지만 그 주변에 Local optimum이 많이 존재하는 함수이다. 

```python
def rastrigin_function(particle_position):
    x, y = particle_position
    A = 10
    return (A * 2 + 
            (x**2 - A * np.cos(2 * np.pi * x)) + 
            (y**2 - A * np.cos(2 * np.pi * y)))

class PSO:
    def __init__(self, objective_func, n_particles, n_dimensions, bounds, w=0.5, c1=1.5, c2=1.5):
        self.objective_func = objective_func
        self.n_particles = n_particles
        self.n_dimensions = n_dimensions
        self.bounds = bounds
        
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.particles_pos = np.random.uniform(bounds[0][0], bounds[0][1], (n_particles, n_dimensions))
        self.particles_vel = np.zeros((n_particles, n_dimensions))
        
        self.pbest_pos = self.particles_pos.copy()
        self.pbest_val = np.array([self.objective_func(p) for p in self.pbest_pos])
        
        min_pbest_idx = np.argmin(self.pbest_val)
        self.gbest_pos = self.pbest_pos[min_pbest_idx].copy()
        self.gbest_val = self.pbest_val[min_pbest_idx]

    def update(self):
        for i in range(self.n_particles):
            r1 = np.random.rand(self.n_dimensions)
            r2 = np.random.rand(self.n_dimensions)
            
            inertia_term = self.w * self.particles_vel[i]
            cognitive_term = self.c1 * r1 * (self.pbest_pos[i] - self.particles_pos[i])
            social_term = self.c2 * r2 * (self.gbest_pos - self.particles_pos[i])
            
            self.particles_vel[i] = inertia_term + cognitive_term + social_term
            
            self.particles_pos[i] += self.particles_vel[i]
            
            self.particles_pos[i] = np.clip(self.particles_pos[i], self.bounds[0][0], self.bounds[0][1])

            current_val = self.objective_func(self.particles_pos[i])
            if current_val < self.pbest_val[i]:
                self.pbest_val[i] = current_val
                self.pbest_pos[i] = self.particles_pos[i].copy()
            
            if current_val < self.gbest_val:
                self.gbest_val = current_val
                self.gbest_pos = self.particles_pos[i].copy()
        
        return self.particles_pos, self.gbest_pos, self.gbest_val
```

#### Phase 2: 최적화 과정 시각화
구현된 PSO 클래스를 이용해 실제로 최적화 과정을 실행하고, Matplotlib을 통해 입자들이 움직이는 모습을 애니메이션으로 그렸다.

```python
N_PARTICLES = 50
N_DIMENSIONS = 2
BOUNDS = [(-5.12, 5.12), (-5.12, 5.12)]
N_ITERATIONS = 150

pso_optimizer = PSO(rastrigin_function, N_PARTICLES, N_DIMENSIONS, BOUNDS)

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(BOUNDS[0])
ax.set_ylim(BOUNDS[1])
ax.set_title('Particle Swarm Optimization')
ax.set_xlabel('x')
ax.set_ylabel('y')

x = np.linspace(BOUNDS[0][0], BOUNDS[0][1], 200)
y = np.linspace(BOUNDS[1][0], BOUNDS[1][1], 200)
X, Y = np.meshgrid(x, y)
Z = rastrigin_function([X, Y])
ax.contour(X, Y, Z, levels=np.linspace(0, 100, 21), cmap='viridis_r') 

particles_scatter = ax.scatter([], [], c='blue', alpha=0.7, label='Particles')
gbest_scatter = ax.scatter([], [], c='red', marker='*', s=200, label='Global Best')
ax.legend()
title_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def animate(i):
    positions, gbest_pos, gbest_val = pso_optimizer.update()
    particles_scatter.set_offsets(positions)
    gbest_scatter.set_offsets(gbest_pos)
    title_text.set_text(f'Iteration: {i+1}, gbest_val: {gbest_val:.4f}')
    return particles_scatter, gbest_scatter, title_text

anim = FuncAnimation(fig, animate, frames=N_ITERATIONS, interval=100, blit=True)
anim.save('pso_rastrigin_animation.gif', writer='pillow', fps=30)
```