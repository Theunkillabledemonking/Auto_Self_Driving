import os
import pandas as pd

# 이미지 폴더 경로
image_folder = r"C:\Users\USER\Desktop\programing\code\data\raw\images"
csv_path = r"C:\Users\USER\Desktop\programing\code\data\raw\training_data.csv"

# 이미지 파일 불러오기
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# 파일명에서 각도 추출 및 CSV 생성
data = []
for file in image_files:
    file_path = os.path.join(image_folder, file)
    # 파일명에서 각도 추출 (예: image_001_angle_90.jpg → 90)
    angle = int(file.split("_angle_")[1].split(".")[0])
    data.append([file_path, angle])

# 데이터프레임 생성 및 CSV 저장
df = pd.DataFrame(data, columns=["frame_path", "angle"])
df.to_csv(csv_path, index=False)
print(f"CSV 파일이 생성되었습니다: {csv_path}")
