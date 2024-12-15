import os
import pandas as pd

# **1. CSV 파일 경로**
csv_path = "C:/Users/USER/Desktop/programing/code/data/oversampled_training_data.csv"

# **2. 데이터 로드**
df = pd.read_csv(csv_path)

# **3. 경로 변경 설정**
old_base_path = "C:/Users/USER/Desktop/programing/code/data/frames/"
new_base_path = "C:/Users/USER/Desktop/programing/code/data/augmented_frames/cropped_"

# 기존 경로를 새로운 경로로 변경
df['frame_path'] = df['frame_path'].str.replace(old_base_path, new_base_path)

# **4. 경로 변경 결과 확인**
print("변경된 경로 예시 (상위 5개):")
print(df['frame_path'].head())

# **5. 수정된 경로 확인 (파일 존재 여부)**
missing_files = []
for path in df['frame_path']:
    if not os.path.exists(path):
        missing_files.append(path)

print(f"누락된 파일 개수: {len(missing_files)}")
if missing_files:
    print(f"누락된 파일 예시: {missing_files[:5]}")
else:
    print("모든 파일이 정상적으로 존재합니다.")

# **6. 변경된 CSV 저장**
updated_csv_path = "C:/Users/USER/Desktop/programing/code/data/training_data_cleaned_updated.csv"
df.to_csv(updated_csv_path, index=False)
print(f"경로가 수정된 CSV를 {updated_csv_path}에 저장했습니다.")
