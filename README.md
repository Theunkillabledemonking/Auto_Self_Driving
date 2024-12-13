project_root/
├── data/                   # 데이터 관련 폴더
│   ├── raw/                # 원본 데이터
│   ├── processed/          # 전처리된 데이터
│   ├── splits/             # 데이터 분할 (train, val, test)
├── scripts/                # 주요 스크립트 폴더
│   ├── main/               # 자동차 제어 관련 코드
│   │   ├── car_control.py  # 메인 자동차 구동 코드
│   │   ├── motor.py        # DC 모터 제어 코드
│   │   ├── servo.py        # 서보 모터 제어 코드
│   │   ├── __init__.py     # Python 패키지 초기화 파일
│   ├── preprocess.py       # 데이터 전처리 코드
│   ├── train.py            # 모델 학습 코드
│   ├── evaluate.py         # 평가 코드
├── tests/                  # 테스트 코드 폴더
│   ├── test_opencv.py      # OpenCV 영상 테스트 코드
│   ├── test_camera.py      # 카메라 연결 확인 및 테스트 코드
├── models/                 # 모델 파일
│   ├── checkpoints/        # 중간 학습 저장 파일
│   ├── final/              # 최종 모델 파일
├── outputs/                # 결과물 폴더
│   ├── logs/               # 로그 파일
│   ├── metrics/            # 평가 결과
│   ├── visualizations/     # 예측 시각화 결과
├── notebooks/              # Jupyter 노트북 파일
│   ├── data_analysis.ipynb # 데이터 분석 노트북
├── README.md               # 프로젝트 설명
├── requirements.txt        # Python 패키지 종속성
├── .gitignore              # Git 제외 파일
