import os
import pandas as pd
import random

# 경로 설정
base_path = r"C:/Users/USER/Desktop/programing/code/data/steering_data_preprocessor"
frame_folder = os.path.join(base_path, "data/frames")  # 이미지 파일이 저장될 폴더
csv_file = os.path.join(base_path, "data/training_data_cleaned.csv")  # 최종 CSV 저장 경로

# 각도 값 리스트
angles = [30, 60, 90, 120, 150]

# 1. 프레임 폴더 확인 및 자동 생성
if not os.path.exists(frame_folder):
    os.makedirs(frame_folder)
    print(f"Frames 폴더를 생성했습니다: {frame_folder}")

# 2. 임시 테스트 이미지 파일 생성 (없을 경우)
test_images = ["image_001.jpg", "image_002.jpg", "image_003.jpg"]
for image in test_images:
    image_path = os.path.join(frame_folder, image)
    with open(image_path, "w") as f:  # 빈 파일 생성
        pass

print(f"테스트 이미지 파일을 생성했습니다: {test_images}")

# 3. 프레임 폴더 내 이미지 파일 읽기
image_files = [f for f in os.listdir(frame_folder) if f.endswith(('.jpg', '.png'))]

if not image_files:
    print("Error: 이미지 파일이 프레임 폴더에 없습니다. 파일을 추가해 주세요.")
else:
    print(f"프레임 폴더에 {len(image_files)}개의 이미지 파일이 있습니다.")

# 4. 이미지 경로와 랜덤 각도 값으로 CSV 데이터 생성
data = []
for image_file in image_files:
    image_path = os.path.join(frame_folder, image_file).replace("\\", "/")  # 경로 표준화
    steering_angle = random.choice(angles)  # 랜덤 각도 할당
    data.append([image_path, steering_angle])

# 데이터프레임 생성 및 저장
df = pd.DataFrame(data, columns=['frame_path', 'steering_angle'])
os.makedirs(os.path.dirname(csv_file), exist_ok=True)  # CSV 폴더 생성
df.to_csv(csv_file, index=False)

print(f"CSV 파일이 생성되었습니다: {csv_file}")
print(df.head())  # 생성된 데이터 확인
