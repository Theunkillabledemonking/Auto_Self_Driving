# **Self-Driving Car 프로젝트**

이 프로젝트는 **Jetson Nano**와 미니 RC카를 활용하여 **라인 트래킹**을 수행하는 자율주행 시스템입니다.  
Jetson Nano의 카메라와 GPIO를 통해 데이터를 수집하고, **PilotNet** 모델을 사용해 주행 경로를 학습 및 제어합니다.

---

## **목차**

1. [프로젝트 개요](#프로젝트-개요)  
2. [폴더 구조](#폴더-구조)  
3. [실행 환경 설정](#실행-환경-설정)  
4. [데이터 전처리](#데이터-전처리)  
5. [모델 학습 및 검증](#모델-학습-및-검증)  
6. [실시간 추론 및 테스트](#실시간-추론-및-테스트)  
7. [결과 및 성능](#결과-및-성능)  

---

## **1. 프로젝트 개요**

- **목적**: Jetson Nano 기반 미니 RC카로 라인 트래킹을 완주하는 자율주행 시스템 구현  
- **사용된 기술**:  
  - **하드웨어**: 카메라, DC 모터, 서보 모터, Jetson Nano  
  - **소프트웨어**:  
    - JetPack 4.6.5  
    - Python 3.6.9  
    - OpenCV 4.1.1  
    - PyTorch 1.10.0  
    - Torchvision 0.11.1  
    - Scikit-learn  

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
│   │   ├── resized_images/     # 리사이즈된 이미지
│   │   └── training_data_resized.csv
│
├── data_preprocessing/         # 데이터 전처리 코드
│   ├── 1image_dataset.py       # 이미지 및 각도 추출
│   ├── 2dataset_path.py        # 데이터 경로 수정
│   ├── 3reset_data.py          # 데이터 검토 및 삭제
│   ├── 4distribution_plot.py   # 조향각 분포 시각화
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

---

## **3. 실행 환경 설정**

## **1. Docker 컨테이너 실행**
Jetson Nano에서 필요한 환경을 Docker로 설정합니다.

````bash
sudo docker run -it \
  --ipc=host \
  --runtime=nvidia \
  --restart=always \
  -v /home/haru/my_project:/workspace \
  --device /dev/video0:/dev/video0 \
  --device /dev/gpiomem:/dev/gpiomem \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --shm-size=1g \
  --privileged \
  --name my_camera_gpio_container \
  ultralytics/ultralytics:latest-jetson-jetpack4 /bin/bash

---

## **2. GPIO 라이브러리 설치**

```basd
pip3 install Cython
git clone https://github.com/NVIDIA/jetson-gpio.git /tmp/jetson-gpio
cd /tmp/jetson-gpio
python3 setup.py install
rm -rf /tmp/jetson-gpio

---

## **3. X11 디스플레이 설정**
호스트 시스템에서 X11 디스플레이 권한을 설정합니다.

```bash
xhost +local:
export DISPLAY=:0.0

---

## **4. 데이터 전처리**

### **1. 이미지 및 각도 추출**
이미지 파일명에서 조향값을 추출하여 CSV 파일로 저장합니다.
````bash
python data_preprocessing/1image_dataset.py

---

## **2. 데이터 경로 수정**
````bash
python data_preprocessing/2dataset_path.py

---

## **3. 불필요한 데이터 검토 및 삭제**
````bash
python data_preprocessing/3reset_data.py

---

## **4. 조향각 분포 시각화**
````bash
python data_preprocessing/4distribution_plot.py

---

## **5. 데이터 오버샘플링**
````bash
python data_preprocessing/5oversample_data.py

---

## **6. 이미지 리사이즈**
````bash
python data_preprocessing/6resize_images.py

---

## **6. 실시간 추론 및 테스트**
학습된 모델을 실시간을 테스트합니다.
python training/train_pilonetr.py

---

## **7. 결과 및 성능**
훈련된 모델 경로: models/best_pilotnet_model.pth
모델 정확도: 약 86% ~ 90%

---
