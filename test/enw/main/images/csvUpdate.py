import os
import pandas as pd
import shutil

# 수정된 경로 설정
current_folder = os.path.dirname(os.path.abspath(__file__))
old_image_folder = os.path.join(current_folder, "imagesV")
new_image_folder = os.path.join(current_folder, "..", "..", "data")
csv_file_path = os.path.join(current_folder, "..", "..", "data", "training_data_cleaned.csv")

# 폴더 생성
os.makedirs(new_image_folder, exist_ok=True)

# 이미지 파일 이동
image_files = [f for f in os.listdir(old_image_folder) if f.endswith(".jpg")]
for image_file in image_files:
    old_path = os.path.join(old_image_folder, image_file)
    new_path = os.path.join(new_image_folder, image_file)
    shutil.move(old_path, new_path)
    print(f"Moved: {old_path} -> {new_path}")

# CSV 파일 수정
if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    df['frame_path'] = df['frame_path'].str.replace("images/imagesV", "data", regex=False)
    df.to_csv(csv_file_path, index=False)
    print(f"CSV 파일이 수정되었습니다: {csv_file_path}")
else:
    print("CSV 파일이 존재하지 않습니다.")
