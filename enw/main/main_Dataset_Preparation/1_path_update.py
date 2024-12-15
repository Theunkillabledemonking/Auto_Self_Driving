import pandas as pd
import os

# 실제 CSV 파일 경로\
csv_path = "C:/Users/USER/Desktop/programing/code/data/rw/training_data.csv"
updated_csv_path = "C:/Users/USER/Desktop/programing/code/data/rw/training_data_updated.csv"
# 디렉토리가 없으면 생성
os.makedirs(os.path.dirname(updated_csv_path), exist_ok=True)

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 경로 수정: "1213"을 "steering_data_preprocessor"로 변경
df['frame_path'] = df['frame_path'].str.replace(
    "1213", "steering_data_preprocessor", regex=False
)

# 수정된 CSV 저장
df.to_csv(updated_csv_path, index=False)

print(f"CSV 경로가 수정되어 저장되었습니다: {updated_csv_path}")
