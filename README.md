
Self-Driving Car 프로젝트
이 프로젝트는 Jetson Nano와 미니 RC카를 활용하여 라인 트래킹을 수행하는 자율주행 시스템입니다. Jetson Nano의 카메라와 GPIO를 통해 데이터를 수집하고, PilotNet 모델을 사용해 주행 경로를 학습 및 제어합니다.

목차
프로젝트 개요
실행 환경 설정
데이터 전처리
모델 학습 및 검증
실시간 추론 및 테스트
결과 및 성능
추가 개선사항
문의 및 기여
프로젝트 개요
목적: Jetson Nano 기반 미니 RC카로 라인 트래킹을 완주하는 자율주행 시스템 구현
사용된 기술
하드웨어: 카메라, DC 모터, 서보 모터, Jetson Nano
소프트웨어:
JetPack 4.6.5
Python 3.6.9
OpenCV 4.1.1
PyTorch 1.10.0
Torchvision 0.11.1
Scikit-learn
실행 환경 설정
1. Docker 컨테이너 실행
Jetson Nano에서 필요한 환경을 Docker로 설정합니다.

bash
코드 복사
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
2. GPIO 라이브러리 설치
컨테이너 내부에서 Jetson GPIO 라이브러리를 설치합니다.

bash
코드 복사
pip3 install Cython
git clone https://github.com/NVIDIA/jetson-gpio.git /tmp/jetson-gpio
cd /tmp/jetson-gpio
python3 setup.py install
rm -rf /tmp/jetson-gpio
3. X11 디스플레이 설정
호스트 시스템에서 X11 권한을 부여합니다.

bash
코드 복사
xhost +local:
export DISPLAY=:0.0
데이터 전처리
데이터 전처리 단계에서 이미지와 조향 각도를 처리하여 모델 학습에 적합한 형태로 가공합니다.

1. 이미지 및 각도 추출
이미지 파일명에서 조향값을 추출하여 CSV 파일로 저장합니다.

bash
코드 복사
python data_preprocessing/1image_dataset.py
코드 예시:

python
코드 복사
# data_preprocessing/1image_dataset.py

import os
import pandas as pd

# 이미지 폴더 경로
image_folder = "data/raw/images"
csv_path = "data/raw/training_data.csv"

# 이미지 파일 목록 가져오기
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
2. 경로 수정 및 누락 데이터 제거
데이터 경로를 수정하고 누락된 이미지를 제거합니다.

bash
코드 복사
python data_preprocessing/2dataset_path.py
코드 예시:

python
코드 복사
# data_preprocessing/2dataset_path.py

import pandas as pd
import os

csv_path = "data/raw/training_data.csv"
updated_csv_path = "data/processed/training_data_updated.csv"

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 경로 수정 (예시: 'old_path'를 'new_path'로 변경)
df['frame_path'] = df['frame_path'].str.replace("old_path", "new_path")

# 존재하지 않는 파일 제거
df = df[df['frame_path'].apply(os.path.exists)]

# 수정된 CSV 저장
df.to_csv(updated_csv_path, index=False)
print(f"수정된 CSV 파일이 저장되었습니다: {updated_csv_path}")
3. 데이터 검토 및 삭제
이미지를 하나씩 확인하여 불필요한 이미지를 삭제합니다.

bash
코드 복사
python data_preprocessing/3reset_data.py
코드 예시:

python
코드 복사
# data_preprocessing/3reset_data.py

import pandas as pd
import cv2
import os

csv_path = "data/processed/training_data_updated.csv"
df = pd.read_csv(csv_path)

index = 0
while True:
    image_path = df.iloc[index]["frame_path"]
    img = cv2.imread(image_path)

    cv2.imshow("Image Viewer", img)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):  # 다음 이미지
        index += 1
    elif key == ord('a'):  # 이전 이미지
        index -= 1
    elif key == ord('c'):  # 이미지 삭제
        os.remove(image_path)
        df = df.drop(index).reset_index(drop=True)
    elif key == ord('q'):  # 종료
        break

cv2.destroyAllWindows()
df.to_csv(csv_path, index=False)
4. 조향각 분포 시각화
조향값의 분포를 그래프로 시각화합니다.

bash
코드 복사
python data_preprocessing/4distribution_steering_angle.py
코드 예시:

python
코드 복사
# data_preprocessing/4distribution_steering_angle.py

import pandas as pd
import matplotlib.pyplot as plt

csv_path = "data/processed/training_data_updated.csv"
df = pd.read_csv(csv_path)

plt.hist(df['angle'], bins=30, color='blue', edgecolor='black')
plt.title("Steering Angle Distribution")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.show()
5. 데이터 오버샘플링
데이터 불균형을 해소하기 위해 오버샘플링을 수행합니다.

bash
코드 복사
python data_preprocessing/5oversample_data.py
코드 예시:

python
코드 복사
# data_preprocessing/5oversample_data.py

import pandas as pd
from sklearn.utils import resample

input_csv = "data/processed/training_data_updated.csv"
output_csv = "data/processed/training_data_oversampled.csv"

df = pd.read_csv(input_csv)
max_count = df['angle'].value_counts().max()

oversampled_data = []
for angle in df['angle'].unique():
    angle_data = df[df['angle'] == angle]
    oversampled = resample(angle_data, replace=True, n_samples=max_count, random_state=42)
    oversampled_data.append(oversampled)

df_oversampled = pd.concat(oversampled_data)
df_oversampled.to_csv(output_csv, index=False)
print(f"오버샘플링 완료: {output_csv}")
6. 이미지 리사이즈
이미지의 상단 20%를 제거하고 (200x66) 크기로 리사이즈합니다.

bash
코드 복사
python data_preprocessing/6resize_images.py
코드 예시:

python
코드 복사
# data_preprocessing/6resize_images.py

import os
import cv2
import pandas as pd

input_csv = "data/processed/training_data_oversampled.csv"
output_csv = "data/processed/training_data_resized.csv"
processed_folder = "data/processed/resized_images"

os.makedirs(processed_folder, exist_ok=True)
df = pd.read_csv(input_csv)
processed_data = []

for _, row in df.iterrows():
    image_path = row['frame_path']
    img = cv2.imread(image_path)

    height = img.shape[0]
    roi = img[int(height * 0.2):, :]
    resized = cv2.resize(roi, (200, 66))

    new_image_path = os.path.join(processed_folder, os.path.basename(image_path))
    cv2.imwrite(new_image_path, resized)
    processed_data.append([new_image_path, row['angle']])

df_resized = pd.DataFrame(processed_data, columns=["frame_path", "angle"])
df_resized.to_csv(output_csv, index=False)
print(f"이미지 리사이즈 완료: {output_csv}")
모델 학습 및 검증
PilotNet 모델을 사용하여 조향값 예측 모델을 학습합니다.

bash
코드 복사
python training/train_pilotnet.py
코드 예시:

python
코드 복사
# training/train_pilotnet.py

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

bash
코드 복사
python testing/test.py
코드 예시:

python
코드 복사
# testing/test.py

import torch
import cv2
from models.pilotnet_model import PilotNet

# 모델 로드
model = PilotNet()
model.load_state_dict(torch.load("models/best_pilotnet_model.pth"))
model.eval()

# 카메라 초기화
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 전처리
    img = cv2.resize(frame, (200, 66))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    # 추론
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    angle = predicted.item()

    # 모터 제어 로직 추가

    # 결과 표시
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
결과 및 성능
훈련된 모델 경로: models/best_pilotnet_model.pth
모델 정확도: 약 86% ~ 90%
추가 개선사항
모델 최적화
