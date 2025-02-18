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


## 데이터 수집
손글씨를 인식하고 마스킹을 해야하므로 손글씨 데이터를 대상으로 웹크롤을 돌렸다.
(파이썬 크롤링 코드)

결국 크롤링은 포기하고 모아져있는 사이트들을 돌아다니던 중 학습용 데이터들을 제공하는 AIHub 라는 사이트를 발견했다. 일차적으로 학습과 검증 단계에서 필요한 데이터들은 AIHub(https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=81) 사이트에서 구했다.
