import os
import cv2
import pandas as pd

# 이미지 파일 경로
image_dir = 'C:/path/to/local/folder/'

# 이미지 파일 목록 가져오기
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# 데이터프레임 생성
data = []

for file in image_files:
    # 파일명에서 메타데이터 추출
    parts = file.split('_')
    timestamp = parts[1] + '_' + parts[2]
    speed = parts[4]
    angle = parts[6].split('.')[0]  # '.jpg' 제거

    # 이미지 로드
    image_path = os.path.join(image_dir, file)
    image = cv2.imread(image_path)

    # 필요한 전처리 수행 (예: 리사이즈)
    # image = cv2.resize(image, (224, 224))

    # 데이터 저장
    data.append({
        'image': image,
        'speed': float(speed),
        'angle': float(angle),
        'timestamp': timestamp
    })

# 데이터프레임 생성
df = pd.DataFrame(data)
