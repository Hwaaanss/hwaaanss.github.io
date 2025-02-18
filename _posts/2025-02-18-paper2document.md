---
layout: single
title: "[Project] Paper to Document"
excerpt: "촬영된 문서를 노트앱에서 수정 가능한 전자 문서로 변환"
mathjax: true
toc: true
toc_sticky: true
toc_label: Code
categories: Dacon
tag: [AI, ML, Project]
---


## 개발 목표
공부를 하다 보면 책이나 공책에 필기를 하고 그것을 사진 찍는 일이 많다. 하지만 나중에 그 사진들을 다시 보면 잘못썼거나, 수정을 하고 싶은 순간이 생긴다. 이런 상황을 위해, 그리고 문서의 디지털화가 본격적으로 일어나는 시대에 맞춰 고안했다. 사진을 찍고 앱에 넣으면 손글씨 부분만 수정 가능한 파일로 변환이 되고, 이를 굿노트와 같은 앱에 넣으면 평소 필기를 수정하듯 수정이 가능하다.


## 개발 순서
전체적인 흐름은 Data Crawling -> Data Preprocessing -> Train the Model -> Model Validation -> Test the Model -> Model Serving(to frontend dev.) 이다. 일단 대략적인 틀만 먼저 잡았고 차차 업데이트 할 예정이다.


## Data Crawling
손글씨를 인식하고 마스킹을 해야하므로 손글씨 데이터를 대상으로 웹크롤을 돌렸다. Java가 지원되는 셀레니움을 사용하기로 했다.

```python
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import urllib.request


query = ['손글씨', '공책 필기', '서적 필기', '문제집 필기','handwriting picture']
num_img = 1000  # 각각 다운할 이미지 개수


# ChromeDriver 지정
driver = webdriver.Chrome()
driver.get("https://www.google.com/imghp?hl=ko&tab=ri&ogbl")
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'MyApp/1.0')]
urllib.request.install_opener(opener)
```

그런데 아무리 해도 썸네일들이 들어오지 않았다. 제대로 클래스 맞춰서 값을 넣었는데 말이다.
한참을 고민하고 시도해본 결과 썸네일의 좌우 가장자리에 흰색으로 레터박스같이 형성된 부분이 있다. 그 부분의 CSS 값이 썸네일의 CSS 값과 다르길래 이걸 넣었더니 정상적으로 이미지들을 긁어오기 시작했다. 거의 3시간의 씨름이었다..

```python
# 검색 및 다운
img_sum = 0     # 전체 이미지 개수 카운트
for iter in range(len(query)):
    elem = driver.find_element(By.NAME, "q")
    elem.clear()
    elem.send_keys(query[iter])
    elem.send_keys(Keys.RETURN)
    time.sleep(5)

    # 이미지 썸네일 리스트 수집
    thums_list =[]
    while True:
        thums_list = driver.find_elements(By.CSS_SELECTOR, '.cC9Rib') 
        if len(thums_list) >= num_img+150:
            print('collected thums_list : {}'.format(len(thums_list)))
            break
        
        more = driver.find_element(By.CSS_SELECTOR, '.GNJvt.ipz2Oe')
        driver.execute_script("arguments[0].click();", more)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
    

    # 리스트에 있는 썸네일들 각각 클릭 후 다운
    cnt = 0     # 각각 다운 성공한 이미지 개수
    for i in range(len(thums_list)):
        try:
            driver.execute_script("arguments[0].click();", thums_list[i])
            time.sleep(1)
            img = driver.find_element(By.XPATH, '//*[@id="Sva75c"]/div[2]/div[2]/div/div[2]/c-wiz/div/div[3]/div[1]/a/img[1]')
            src = img.get_attribute('src')    
            urllib.request.urlretrieve(src, f"./images/train_{(n (다운 진행상황에 맞춰 0 이나 500 으로)+img_sum):04d}.png")
            cnt += 1
            img_sum += 1
            time.sleep(1)
            
        except Exception as e:
            print(f"Error downloading image {i}: {e}")
        
        if cnt >= num_img:
            break
        
time.sleep(2)
```

하지만 크롤링 후 정제를 해보니 몇번을 다시 다운 받거나, 이미지가 학습에 적합하지 않은 상황이 많이 만들어져서 다른 방법을 찾기로 했다.
그러다가 학습용 데이터들을 제공하는 AIHub 라는 사이트를 발견했다. 일차적으로 학습과 검증 단계에서 필요한 데이터들은 AIHub(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=81) 사이트에서 구했다. 흰 배경에 검정색으로 손글씨를 써 놓은 이미지들이다.


## Data Preprocessing
전처리에는 다양한 기법들이 있고, 이는 데이터의 특성에 따라 다양한 기법들을 적절히 사용해야 한다. 데이터 가공을 잘 하기만 해도 모델의 성능이 훨씬 올라가기 때문이다.

#### 나중에 혹시나 직접 찍은 사진들을 훈련용 데이터로 쓸 일이 있을까봐 아이폰에서 사용되는 포맷인 heif 파일을 열기 위해 라이브러리를 미리 준비해놨다.
```python
import os
import shutil
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pillow_heif import register_heif_opener
```

#### 경로들을 먼저 구분해줬다. 코드 안에 하드코딩 해두는 것보다는 이 방식이 추후 수정에 편리할 것 같아서다.
```python
# Directory setting (지속적인 성능 업데이트를 위해 추가 수집된 학습용 raw data 는 dump로 들어가고, resize 한 후 processed 폴더로 이동)
train_dump_dir = './train/dump'
val_dump_dir = './validation/dump'

train_processed_dir = './train/processed'
val_processed_dir = './validation/processed'

# dump -> img 폴더의 중복 방지를 위한 파일명 리스트
train_dump_fns = sorted(os.listdir(train_dump_dir))
train_processed_fns = sorted(os.listdir(train_processed_dir))

val_dump_fns = sorted(os.listdir(val_dump_dir))
val_processed_fns = sorted(os.listdir(val_processed_dir))
```

#### dump, processed 각 폴더에 들어있는 데이터의 개수 파악
```python
def CheckData():
    print('number of dump data (train, val): ',len(train_dump_fns), len(val_dump_fns))
    print('number of processed data (train, val): ',len(train_processed_fns), len(val_processed_fns))
    # sample_img = Image.open(os.path.join(train_processed_dir, train_processed_fns[0]))
    # sample_img.show()
    # arr1 = np.array(sample_img)
    # print('processed image shape: ',arr1.shape)  # h,w,c format
```

#### 전처리 아예 안된 공책 필기 데이터를 dump -> img 이동 시 이진화, 사이즈 조절, 이름 변경 적용된 함수 / 임시로 train 대상으로 해놓음
위에서 heif 이미지 파일을 대비한 것과 같은 이유로 준비했으나, 수정이 필요한 부분이다. resize 대신 슬라이드 윈도우로 자르는 방식이 맞을 듯 싶다.
```python
def AdjustData():
    register_heif_opener()
    for i in range(len(train_dump_fns)):
        train_tmp = Image.open(os.path.join(train_dump_dir, train_dump_fns[i]))
        train_tmp_cv = cv2.cvtColor(np.array(train_tmp), cv2.COLOR_BGR2GRAY)
        _, train_tmp_cv = cv2.threshold(train_tmp_cv, 128, 255, cv2.THRESH_BINARY)
        resized_img = cv2.resize(train_tmp_cv, (512, 512))
        save_path = os.path.join(train_processed_dir, "train_{}.png".format(i+len(train_processed_fns)))
        cv2.imwrite(save_path, resized_img)
```     

#### 훈련용 문장 데이터를 세로로 20장씩 이어 붙힌 후 3584x3584 로 사이즈 통일
```python
def TConcatsplit():
    for i in range(0,len(train_dump_fns),20):
        img1 = np.array(Image.open(os.path.join(train_dump_dir, train_dump_fns[i])))
        new_img = cv2.resize(img1, (3584, 179))
        for ii in range(1,20):
            img2 = np.array(Image.open(os.path.join(train_dump_dir, train_dump_fns[i+ii])))
            img2 = cv2.resize(img2, (3584, 179))
            new_img = cv2.vconcat([new_img, img2])
        new_img = cv2.resize(new_img, (3584, 3584))
        for k in range(7):
            for kk in range(7):
                sliced_img = new_img[512*k:512*(k+1), 512*kk:512*(kk+1)]
                train_processed_fns = sorted(os.listdir(os.path.join(train_processed_dir, 'printed')))
                cv2.imwrite(os.path.join(train_processed_dir,'printed', 'train_{}.png'.format(len(train_processed_fns))), sliced_img)
```

#### 검증용 문장 데이터를 세로로 20장씩 이어 붙힌 후 3584x3584 로 사이즈 통일
```python
def VConcatsplit():
    for i in range(0,len(val_dump_fns),20):
        img1 = np.array(Image.open(os.path.join(val_dump_dir, val_dump_fns[i])))
        new_img = cv2.resize(img1, (3584, 179))
        for ii in range(1,20):
            img2 = np.array(Image.open(os.path.join(val_dump_dir, val_dump_fns[i+ii])))
            img2 = cv2.resize(img2, (3584, 179))
            new_img = cv2.vconcat([new_img, img2])
        new_img = cv2.resize(new_img, (3584, 3584))
        for k in range(7):
            for kk in range(7):
                sliced_img = new_img[512*k:512*(k+1), 512*kk:512*(kk+1)]
                val_processed_fns = sorted(os.listdir(os.path.join(val_processed_dir, 'printed')))
                cv2.imwrite(os.path.join(val_processed_dir,'printed', 'val_{}.png'.format(len(val_processed_fns))), sliced_img)
```

#### 위의 Concat 과 Split 전에 해야 하는 작업으로, 길이가 너무 짧은 이미지들을 제거한다. (검수 결과 간혹 가로가 터무니 없는 비율의 이미지들이 있었음)
```python
def RemoveShort():
    for i in range(len(train_dump_fns)):
        if train_dump_fns[i] == '.DS_Store':
            os.remove(os.path.join(train_dump_dir, train_dump_fns[i]))
        else:
            img = np.array(Image.open(os.path.join(train_dump_dir, train_dump_fns[i])))
            if img.shape[1] <= 3000:
                os.remove(os.path.join(train_dump_dir, train_dump_fns[i]))
```

```python
RemoveShort()
TConCatsplit()
VConCatsplit()
```

## Train the Model
