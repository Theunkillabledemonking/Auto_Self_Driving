import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로 수정
csv_path = r"data/processed/training_data_updated.csv"

try:
    # CSV 파일 로드
    df = pd.read_csv(csv_path)

    # 조향각 분포 시각화 (열 이름 'angle' 사용)
    plt.figure(figsize=(10, 6))
    plt.hist(df['angle'], bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {csv_path}")
except KeyError as e:
    print(f"열 이름 오류: {e}")
except Exception as e:
    print(f"오류 발생: {e}")
