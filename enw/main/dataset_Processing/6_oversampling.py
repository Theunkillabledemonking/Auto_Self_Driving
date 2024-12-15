import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample

# **1. CSV 파일 경로**
csv_path = "C:/Users/USER/Desktop/programing/code/data/training_data_cleaned.csv"

# **2. CSV 파일 로드**
df = pd.read_csv(csv_path)

# **3. 각 클래스별 데이터 나누기**
class_30 = df[df['steering_angle'] == 30]
class_60 = df[df['steering_angle'] == 60]
class_90 = df[df['steering_angle'] == 90]
class_120 = df[df['steering_angle'] == 120]

# **4. Oversampling 및 Undersampling 적용**
# 소수 클래스 Oversampling (1000개로 맞춤)
class_30_oversampled = resample(class_30, replace=True, n_samples=1000, random_state=42)
class_120_oversampled = resample(class_120, replace=True, n_samples=1000, random_state=42)

# 다수 클래스 Undersampling (1000개로 맞춤)
class_60_undersampled = class_60.sample(n=1000, random_state=42)
class_90_undersampled = class_90.sample(n=1000, random_state=42)

# **5. 최종 데이터셋 병합**
df_balanced = pd.concat([
    class_30_oversampled,
    class_60_undersampled,
    class_90_undersampled,
    class_120_oversampled
])

# **6. 결과 저장 경로**
balanced_csv_path = "C:/Users/USER/Desktop/programing/code/data/balanced_training_data.csv"
df_balanced.to_csv(balanced_csv_path, index=False)
print(f"균형 잡힌 데이터셋이 저장되었습니다: {balanced_csv_path}")

# **7. 데이터 분포 시각화**
plt.figure(figsize=(10, 6))
plt.hist(df_balanced['steering_angle'], bins=30, color='purple', alpha=0.7, edgecolor='black')
plt.title("Balanced Steering Angle Distribution")
plt.xlabel("Steering Angle")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
