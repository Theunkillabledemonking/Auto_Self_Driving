# **Self-Driving Car 프로젝트**

![Project Logo](https://your-logo-url.com/logo.png)

이 프로젝트는 **Jetson Nano**와 미니 RC카를 활용하여 **라인 트래킹**을 수행하는 자율주행 시스템입니다.  
Jetson Nano의 카메라와 GPIO를 통해 데이터를 수집하고, **PilotNet** 모델을 사용해 주행 경로를 학습 및 제어합니다.

---

## **목차**

1. [프로젝트 개요](#1-프로젝트-개요)
2. [폴더 구조](#2-폴더-구조)
3. [시작 가이드](#3-시작-가이드)
4. [기술 스택](#4-기술-스택)
5. [주요 기능](#5-주요-기능)
6. [결과 및 성능](#6-결과-및-성능)
7. [기타 추가 사항들](#7-기타-추가-사항들)
8. [개발자 소개](#8-개발자-소개)
9. [추가 팁](#9-추가-팁)

---

## **1. 프로젝트 개요**

- **프로젝트 이름**: Self-Driving Car 프로젝트
- **목적**: Jetson Nano 기반 미니 RC카로 라인 트래킹을 완주하는 자율주행 시스템 구현
- **개발 기간**: 2023년 10월 ~ 2024년 12월 18일
- **사용된 기술**:
  - **하드웨어**: 카메라, DC 모터, 서보 모터, Jetson Nano
  - **소프트웨어**:
    - JetPack 4.6.5
    - Python 3.6.9
    - OpenCV 4.1.1
    - PyTorch 1.10.0
    - Torchvision 0.11.1
    - Scikit-learn
    - CUDA

---

## **2. 폴더 구조**

```plaintext
Self-Driving-Car-Project/
│
├── data/                       # 데이터 디렉터리
│   ├── raw/                    # 원본 데이터
│   │   ├── images/             # 이미지 파일
│   │   └── training_data.csv   # 조향각 데이터
│   ├── processed/              # 전처리된 데이터
│       ├── resized_images/     # 리사이즈된 이미지
│       └── training_data_resized.csv
│
├── data_preprocessing/         # 데이터 전처리 코드
│   ├── 1image_dataset.py       # 이미지 및 각도 추출
│   ├── 2dataset_path.py        # 데이터 경로 수정
│   ├── 3reset_data.py          # 데이터 검토 및 삭제
│   ├── 4distribution_plot.py   # 조향값 분포 시각화
│   ├── 5oversample_data.py     # 데이터 오버샘플링
│   └── 6resize_images.py       # 이미지 리사이즈
│
├── models/                     # 모델 관련 디렉터리
│   └── best_pilotnet_model.pth # 학습된 모델
│
├── training/                   # 학습 코드
│   └── train_pilotnet.py       # PilotNet 모델 학습 코드
│
├── testing/                    # 실시간 테스트 코드
│   └── test.py                 # 실시간 추론 및 테스트
│
├── README.md                   # 프로젝트 설명 파일
└── requirements.txt            # 의존성 패키지 목록
3. 시작 가이드
01. 프로젝트에 대한 정보
(1) 프로젝트 이름
Self-Driving Car 프로젝트

(2) 프로젝트 로고나 이미지

(3) 프로젝트 소개
Jetson Nano와 미니 RC카를 활용하여 라인 트래킹을 수행하는 자율주행 시스템입니다. 카메라와 GPIO를 통해 데이터를 수집하고, PilotNet 모델을 사용해 주행 경로를 학습 및 제어합니다.

(4) 배포주소
프로젝트의 최종본은 배포주소를 통해 확인할 수 있습니다.

(5) 개발기간
2023년 10월 ~ 2024년 12월 18일

(6) 개발자 소개
위 개발자 소개 섹션을 참고하세요.

02. 시작 가이드
(1) 요구 사항
프로젝트를 클론하여 실행하기 위해 필요한 요구 사항과 버전은 requirements.txt 파일을 참고하세요.

Docker: 최신 버전
Python: 3.6.9
JetPack: 4.6.5
CUDA: 버전에 맞는 설치 필요
(2) 설치 및 실행
Repository 클론하기

git clone https://github.com/theunkillabledemonking/Self-Driving-Car-Project.git
cd Self-Driving-Car-Project
필요한 패키지 설치하기

pip3 install -r requirements.txt
환경 변수 설정

필요한 환경 변수를 .env 파일에 설정합니다.

프로젝트 실행하기

python main.py
4. 기술 스택
프로젝트에 사용된 주요 기술 스택은 다음과 같습니다.


더 많은 기술 스택이 필요하다면 Shields.io에서 원하는 배지를 생성할 수 있습니다!

5. 주요 기능
라인 트래킹: RC카가 지정된 라인을 따라 정확하게 주행합니다.
실시간 데이터 수집: Jetson Nano의 카메라와 GPIO를 통해 주행 데이터를 실시간으로 수집합니다.
모델 학습: PilotNet 모델을 사용하여 주행 경로를 학습합니다.
실시간 추론: 학습된 모델을 기반으로 실시간으로 주행 경로를 제어합니다.
데이터 전처리: 이미지 리사이즈, 오버샘플링 등 다양한 데이터 전처리 과정을 포함합니다.
모델 검증: 학습된 모델의 정확도를 검증하여 최적의 성능을 보장합니다.
6. 결과 및 성능
훈련된 모델 경로: models/best_pilotnet_model.pth
모델 정확도: 약 86% ~ 90%
7. 개발자 소개
이 프로젝트는 theunkillabledemonking에 의해 단독으로 개발되었습니다.

<table> <tbody> <tr> <td align="center"> <a href="https://github.com/theunkillabledemonking"> <img src="https://avatars.githubusercontent.com/u/your-username?v=4" width="100px;" alt="theunkillabledemonking"/><br /> <sub><b>theunkillabledemonking</b></sub> </a><br /> 개발자 </td> </tr> </tbody> </table>
