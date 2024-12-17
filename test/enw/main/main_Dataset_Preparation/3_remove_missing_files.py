import pandas as pd
import os

# 수정된 CSV 파일 경로
csv_path = "C:/Users/USER/Desktop/programing/code/data/training_data_updated.csv"

# 기준 경로 설정 (이 경로는 frame_path 열의 경로 앞에 붙여져야 합니다)
base_path = "C:/Users/USER/Desktop/programing/code/data/"

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 존재하지 않는 파일 확인 (상대 경로를 기준 경로와 결합하여 절대 경로로 변환)
missing_files = [os.path.join(base_path, path) for path in df['frame_path'] if not os.path.exists(os.path.join(base_path, path))]

# 결과 출력
print(f"최종 누락된 파일 개수: {len(missing_files)}")
if len(missing_files) > 0:
    print(f"누락된 파일 예시: {missing_files[:5]}")  # 누락된 파일 중 5개 예시 출력
else:
    print("모든 파일이 존재합니다.")
