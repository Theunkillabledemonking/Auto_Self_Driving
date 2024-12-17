import pandas as pd
import os

# Original CSV file path
csv_path = "C:/Users/USER/Desktop/programing/code/data/training_data_updated.csv"

# Load the CSV file
df = pd.read_csv(csv_path)

# 경로를 절대 경로로 변환 (파일 경로가 상대 경로일 경우)
base_path = "C:/Users/USER/Desktop/programing/code/data/"  # 기준 경로

# Check for missing files
missing_files = [os.path.join(base_path, path) for path in df['frame_path'] if not os.path.exists(os.path.join(base_path, path))]

# Print results
print(f"Number of missing files: {len(missing_files)}")
if len(missing_files) > 0:
    print(f"Example missing files: {missing_files[:5]}")  # 처음 5개 예시 출력
else:
    print("All files exist.")
