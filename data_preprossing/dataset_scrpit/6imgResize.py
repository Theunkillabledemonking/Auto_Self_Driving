import os
import cv2
import pandas as pd

# 경로 설정 (상대 경로 사용)
input_csv = os.path.join("data", "processed", "training_data_oversampled.csv")
processed_folder = os.path.join("data", "processed", "resized_images")
output_csv = os.path.join("data", "processed", "training_data_resized.csv")

# 출력 폴더 생성
os.makedirs(processed_folder, exist_ok=True)

try:
    # CSV 파일 로드
    df = pd.read_csv(input_csv)
    print(f"데이터 로드 완료: {input_csv}")

    processed_data = []
    for _, row in df.iterrows():
        image_path = row['frame_path']
        if os.path.exists(image_path):
            # 이미지 로드
            img = cv2.imread(image_path)

            # 상단 20% 제거 및 리사이즈 (200x66)
            height, width = img.shape[:2]
            roi = img[int(height * 0.2):, :]  # 상단 20% 제거
            resized = cv2.resize(roi, (200, 66))  # 리사이즈

            # 새 이미지 저장
            new_filename = os.path.basename(image_path).replace(".jpg", "_resized.jpg")
            new_image_path = os.path.join(processed_folder, new_filename)
            cv2.imwrite(new_image_path, resized)

            # 데이터 기록
            processed_data.append([new_image_path, row['angle']])
        else:
            print(f"이미지를 찾을 수 없습니다: {image_path}")

    # 새 CSV 저장
    df_resized = pd.DataFrame(processed_data, columns=["frame_path", "angle"])
    df_resized.to_csv(output_csv, index=False)
    print(f"이미지 리사이즈 완료. 저장된 경로: {output_csv}")

except FileNotFoundError:
    print(f"파일을 찾을 수 없습니다: {input_csv}")
except Exception as e:
    print(f"오류 발생: {e}")
