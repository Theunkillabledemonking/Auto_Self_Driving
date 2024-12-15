project_root/
├── data/                   # 데이터 관련 폴더
│   ├── raw/                # 원본 데이터
│   ├── processed/          # 전처리된 데이터
│   ├── splits/             # 데이터 분할 (train, val, test)
│
├── scripts/                # 주요 스크립트 폴더
│   ├── main/               # 자동차 제어 관련 코드
│   │   ├── car_control.py  # 자동차 구동 메인 코드
│   │   ├── motor.py        # DC 모터 제어 코드
│   │   ├── servo.py        # 서보 모터 제어 코드
│   │   ├── __init__.py     # 패키지 초기화 파일
│   ├── preprocess.py       # 데이터 전처리 코드
│   ├── train.py            # 모델 학습 코드
│   ├── evaluate.py         # 모델 평가 코드
│
├── tests/                  # 테스트 코드 폴더
│   ├── test_opencv.py      # OpenCV 테스트 코드
│   ├── test_camera.py      # 카메라 연결 테스트 코드
│
├── models/                 # 모델 파일 저장 폴더
│   ├── checkpoints/        # 학습 중간 저장 파일
│   ├── final/              # 최종 모델 파일
