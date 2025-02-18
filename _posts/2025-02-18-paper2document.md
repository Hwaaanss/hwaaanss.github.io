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
내가 준비한 데이터는 3578