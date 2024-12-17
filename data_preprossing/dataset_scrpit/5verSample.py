import pandas as pd
from sklearn.utils import resample
import os
import matplotlib.pyplot as plt

# CSV 파일 경로
input_csv = os.path.join("data", "processed", "training_data_updated.csv")
output_csv = os.path.join("data", "processed", "training_data_oversampled.csv")

try:
    # CSV 파일 로드
    df = pd.read_csv(input_csv)
    print(f"데이터 로드 완료: {input_csv}")

    # 최대 데이터 수 확인
    max_count = df['angle'].value_counts().max()
    print(f"최대 데이터 수: {max_count}")

    # 오버샘플링
    oversampled_data = []
    for angle in df['angle'].unique():
        angle_data = df[df['angle'] == angle]
        oversampled = resample(angle_data, replace=True, n_samples=max_count, random_state=42)
        oversampled_data.append(oversampled)

    # 병합 및 저장
    df_oversampled = pd.concat(oversampled_data)
    df_oversampled.to_csv(output_csv, index=False)
    print(f"오버샘플링 완료. 저장된 경로: {output_csv}")

    # **시각화: 오버샘플링 전후 데이터 분포**
    plt.figure(figsize=(12, 6))

    # 원본 데이터 분포
    plt.subplot(1, 2, 1)
    df['angle'].value_counts().sort_index().plot(kind='bar', color='blue', alpha=0.7, edgecolor='black')
    plt.title("Original Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")

    # 오버샘플링 후 데이터 분포
    plt.subplot(1, 2, 2)
    df_oversampled['angle'].value_counts().sort_index().plot(kind='bar', color='green', alpha=0.7, edgecolor='black')
    plt.title("Oversampled Steering Angle Distribution")
    plt.xlabel("Steering Angle")
    plt.ylabel("Frequency")

    # 그래프 출력
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {input_csv}")
except Exception as e:
    print(f"오류 발생: {e}")
