
# 🚗 **Self-Driving Car 프로젝트**

![Self-Driving Car](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Self_Driving_Car.jpg/640px-Self_Driving_Car.jpg)

이 프로젝트는 **Jetson Nano**와 미니 RC카를 활용하여 **라인 트래킹**을 수행하는 자율주행 시스템입니다.  
Jetson Nano의 카메라와 GPIO를 통해 데이터를 수집하고, **PilotNet** 모델을 사용해 주행 경로를 학습 및 제어합니다.

---

## 🗂️ **목차**

1. [프로젝트 개요](#1-프로젝트-개요)  
2. [폴더 구조](#2-폴더-구조)  
3. [시작 가이드](#3-시작-가이드)  
4. [기술 스택](#4-기술-스택)  
5. [주요 기능](#5-주요-기능)  
6. [결과 및 성능](#6-결과-및-성능)  
7. [개발자 소개](#7-개발자-소개)  
8. [추가 팁](#8-추가-팁)  

---

## 🚀 **1. 프로젝트 개요**

- **프로젝트 이름**: Self-Driving Car 프로젝트  
- **목적**: **Jetson Nano** 기반 미니 RC카로 라인 트래킹을 완주하는 자율주행 시스템 구현  
- **개발 기간**: 📅 2024년 **9월** ~ 2024년 **12월 18일**  
- **사용된 기술**:
  - 🛠️ **하드웨어**: 카메라, DC 모터, 서보 모터, **Jetson Nano**  
  - 💻 **소프트웨어**:  
    - JetPack 4.6.5  
    - Python 3.6.9  
    - OpenCV 4.1.1  
    - PyTorch 1.10.0  
    - Torchvision 0.11.1  
    - Scikit-learn  
    - CUDA 10.2

---

## 📁 **2. 폴더 구조**

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
```

---

## ⚙️ **3. 시작 가이드**

### ▶️ **3.1 설치 및 실행**

#### **1) 프로젝트 클론**
먼저 GitHub 저장소를 클론합니다.

```bash
git clone https://github.com/theunkillabledemonking/Self-Driving-Car-Project.git
cd Self-Driving-Car-Project
```

#### **2) Docker 환경 설정**
Docker를 사용해 컨테이너를 실행하려면 아래 명령어를 사용하거나, `docker-compose.yml` 파일을 활용하세요.

##### **Bash 명령어**
```bash
sudo docker run -it \
  --ipc=host \
  --runtime=nvidia \
  --restart=always \
  -v $(pwd):/workspace \               # 로컬 디렉토리 연결
  --device /dev/video0:/dev/video0 \   # 카메라 디바이스 연결
  --device /dev/gpiomem:/dev/gpiomem \ # GPIO 디바이스 연결
  -e DISPLAY=$DISPLAY \                # X11 디스플레이 환경 변수 전달
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \   # X11 유닉스 소켓 연결
  --shm-size=1g \                      # 공유 메모리 크기 설정
  --privileged \                       # 특권 모드 활성화
  --name my_camera_gpio_container \    # 컨테이너 이름
  ultralytics/ultralytics:latest-jetson-jetpack4 /bin/bash
```

##### **Docker Compose 사용**
`docker-compose.yml` 파일을 생성하거나, 아래 내용을 프로젝트 디렉토리에 추가하세요.

```yaml
version: "3.8"
services:
  my_camera_gpio_service:
    image: ultralytics/ultralytics:latest-jetson-jetpack4
    container_name: my_camera_gpio_container
    runtime: nvidia
    ipc: host
    privileged: true
    restart: always
    shm_size: 1g
    volumes:
      - ./workspace:/workspace
      - /tmp/.X11-unix:/tmp/.X11-unix
    devices:
      - /dev/video0:/dev/video0
      - /dev/gpiomem:/dev/gpiomem
    environment:
      DISPLAY: $DISPLAY
      QT_X11_NO_MITSHM: "1"
```

Docker Compose로 실행하려면 아래 명령어를 사용하세요:

```bash
docker-compose up -d
```

#### **3) 컨테이너 접속 및 실행**
컨테이너 실행 후 아래 명령어로 내부에 접속하여 프로젝트를 실행하세요.

```bash
docker exec -it my_camera_gpio_container /bin/bash
python train_pilotnet.py
```

---

### ⚙️ **추가 설정**

#### **X11 디스플레이 활성화**
```bash
xhost +local:docker
```

#### **Jetson 장치 확인**
NVIDIA Jetson 환경에서 CUDA와 JetPack 버전이 호환되는지 확인하세요.

#### **환경 변수 설정**
`.env` 파일을 사용해 환경 변수를 정의하고 컨테이너에 전달할 수 있습니다.

---

## 🛠️ **4. 기술 스택**

| 🚀 **기술**       | **설명**                        |
|------------------|--------------------------------|
| **Jetson Nano**  | Edge AI 하드웨어 플랫폼          |
| **OpenCV**       | 이미지 처리 및 분석 라이브러리    |
| **PyTorch**      | 딥러닝 모델 학습 및 추론        |
| **Torchvision**  | 이미지 데이터셋 지원 도구       |
| **Scikit-learn** | 데이터 전처리 및 분석            |
| **CUDA**         | GPU 기반 컴퓨팅 지원            |

---

## 🌟 **5. 주요 기능**

1. **라인 트래킹**: RC카가 지정된 라인을 따라 주행  
2. **실시간 데이터 수집**: 카메라 및 GPIO를 통해 데이터 수집  
3. **모델 학습**: PilotNet 모델을 학습하여 주행 경로 예측  
4. **실시간 추론**: 학습된 모델을 기반으로 실시간 주행 제어  
5. **데이터 전처리**: 이미지 리사이즈 및 데이터 증강 수행  
6. **모델 검증**: 성능 검증 및 최적화  

---

## 📊 **6. 결과 및 성능**

- **훈련된 모델 경로**: `models/best_pilotnet_model.pth`  
- **모델 정확도**: 약 **86% ~ 90%**  

---

## 👤 **7. 개발자 소개**

이 프로젝트는 **theunkillabledemonking**에 의해 개발되었습니다.  

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/theunkillabledemonking">
        <img src="https://avatars.githubusercontent.com/u/166793267?v=4" width="100px;" alt="theunkillabledemonking"/>
        <br />
        <sub><b>theunkillabledemonking</b></sub>
      </a>
      <br />
      🚗 개발자
    </td>
  </tr>
</table>

---

## 💡 **8. 추가 팁**

- **Jetson Nano**에 최적화된 소프트웨어를 설치해주세요.
- **PilotNet** 모델의 구조를 변경하여 추가 성능을 향상할 수 있습니다.  
- 데이터 수집 시 충분한 주행 데이터를 확보하는 것이 중요합니다.

---

🚀 **성공적인 프로젝트 진행을 기원합니다!** 🚗💨
