import pandas as pd
import os

csv_path = r"C:\Users\USER\Desktop\programing\code\data\raw\training_data.csv"
updated_csv_path = r"C:\Users\USER\Desktop\programing\code\data\processed\training_data_updated.csv"

# 수정된 디렉토리가 존재하지 않으면 생성
os.makedirs(os.path.dirname(updated_csv_path), exist_ok=True)

# CSV 파일 불러오기
df = pd.read_csv(csv_path)

# 1. 경로 수정: "1213" → "steering_data_preprocessor"
df['frame_path'] = df['frame_path'].str.replace("1213", "steering_data_preprocessor", regex=False)
print("경로 수정 완료: '1213' → 'steering_data_preprocessor'")

# 2. 존재하지 않는 파일 확인 및 제거
missing_files = []
for idx, path in enumerate(df['frame_path']):
    if not os.path.exists(path):
        missing_files.append(path)

if missing_files:
    print("\n누락된 파일 목록:")
    for file in missing_files:
        print(file)
    # 누락된 파일 제거
    df = df[~df['frame_path'].isin(missing_files)]
    print(f"\n{len(missing_files)}개의 누락된 파일이 제거되었습니다.")

# 3. 불필요한 열 제거: 'direction' 열 확인 및 삭제
if "direction" in df.columns:
    df = df.drop(columns=["direction"])
    print("불필요한 'direction' 열이 제거되었습니다.")

# 4. 수정된 CSV 파일 저장
df.to_csv(updated_csv_path, index=False)
print(f"\n수정된 CSV 파일이 저장되었습니다: {updated_csv_path}")
