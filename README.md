# **Self-Driving Car 프로젝트**

이 프로젝트는 **Jetson Nano**와 미니 RC카를 활용하여 **라인 트래킹**을 수행하는 자율주행 시스템입니다.  
Jetson Nano의 카메라와 GPIO를 통해 데이터를 수집하고, **PilotNet** 모델을 사용해 주행 경로를 학습 및 제어합니다.

---

## **목차**

1. [프로젝트 개요](#프로젝트-개요)
2. [실행 환경 설정](#실행-환경-설정)
3. [데이터 전처리](#데이터-전처리)
4. [모델 학습 및 검증](#모델-학습-및-검증)
5. [실시간 추론 및 테스트](#실시간-추론-및-테스트)
6. [결과 및 성능](#결과-및-성능)

---

## **프로젝트 개요**

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

## **실행 환경 설정**

### **1. Docker 컨테이너 실행**

Jetson Nano에서 필요한 환경을 Docker로 설정합니다.

```bash
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

### **1. GPIO 라이브러리 설치**
컨테이너 내부에서 Jetson GPIO 라이브러리를 설치합니다.

````bash
pip3 install Cython
git clone https://github.com/NVIDIA/jetson-gpio.git /tmp/jetson-gpio
cd /tmp/jetson-gpio
python3 setup.py install
rm -rf /tmp/jetson-gpio

---

### **3. X11 디스플레이 설정**
호스트 시스템에서 X11 디스플레이 권한을 설정합니다.

````bash
xhost +local:
export DISPLAY=:0.0

---

데이터 전처리
1. 이미지 및 각도 추출
이미지 파일명에서 조향값을 추출하여 CSV 파일로 저장합니다.

python data_preprocessing/1image_dataset.py
코드 예시:

import os
import pandas as pd

# 이미지 폴더 경로
image_folder = "data/raw/images"
csv_path = "data/raw/training_data.csv"

# 이미지 파일 불러오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# 파일명에서 각도 추출 및 CSV 생성
data = []
for file in image_files:
    file_path = os.path.join(image_folder, file)
    angle = int(file.split("_angle_")[1].split(".")[0])
    data.append([file_path, angle])

# 데이터프레임 생성 및 CSV 저장
df = pd.DataFrame(data, columns=["frame_path", "angle"])
df.to_csv(csv_path, index=False)
print(f"CSV 파일이 생성되었습니다: {csv_path}")
모델 학습 및 검증
PilotNet 모델 학습
PilotNet 모델을 사용하여 조향값을 예측하는 모델을 학습합니다.

python training/train_pilotnet.py
코드 예시:

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.pilotnet_model import PilotNet
from datasets.steering_dataset import SteeringDataset

# 데이터셋 로드
train_dataset = SteeringDataset("data/processed/training_data_resized.csv")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 모델 초기화
model = PilotNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# 학습 루프
for epoch in range(30):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/30], Loss: {loss.item():.4f}")

# 모델 저장
torch.save(model.state_dict(), "models/best_pilotnet_model.pth")
실시간 추론 및 테스트
학습된 모델을 사용하여 실시간으로 RC카를 제어합니다.

python testing/test.py
결과 및 성능
훈련된 모델 경로: models/best_pilotnet_model.pth
모델 정확도: 약 86% ~ 90%
