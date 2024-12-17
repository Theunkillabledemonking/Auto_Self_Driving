import os
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_csv_from_images(image_folder, output_csv):
    """
    이미지 파일 이름에서 각도(angle) 정보를 추출하여 CSV 파일을 생성합니다.
    
    Args:
        image_folder (str): 이미지 파일이 저장된 폴더 경로
        output_csv (str): 생성될 CSV 파일의 경로
    """
    # 정규 표현식: angle_뒤에 오는 숫자만 추출
    angle_pattern = re.compile(r'angle_(\d+)')

    # 데이터를 저장할 리스트
    data = []

    # 이미지 폴더 내의 파일 순회
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            match = angle_pattern.search(filename)
            if match:
                angle = int(match.group(1))
                data.append([filename, angle])  # 이미지 이름과 각도 기록
            else:
                print(f"Invalid angle format in filename: {filename}")

    # CSV 파일 생성
    df = pd.DataFrame(data, columns=["image_name", "steering_angle"])
    df.to_csv(output_csv, index=False)
    print(f"CSV 파일 생성 완료: {output_csv}")

    return df


def split_dataset(df, output_dir):
    """
    데이터프레임을 train, val, test로 나누어 CSV 파일로 저장합니다.
    
    Args:
        df (pd.DataFrame): 전체 데이터셋
        output_dir (str): 나뉜 데이터셋을 저장할 폴더 경로
    """
    # 데이터셋 나누기: 70% 훈련, 15% 검증, 15% 테스트
    train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # 나눈 데이터를 저장
    train_data.to_csv(os.path.join(output_dir, "train_labels.csv"), index=False)
    val_data.to_csv(os.path.join(output_dir, "val_labels.csv"), index=False)
    test_data.to_csv(os.path.join(output_dir, "test_labels.csv"), index=False)

    print("데이터셋 나누기 완료:")
    print(f"Train: {len(train_data)}개")
    print(f"Validation: {len(val_data)}개")
    print(f"Test: {len(test_data)}개")


# 실행 예제
if __name__ == "__main__":
    image_folder = "data/processed/balanced"  # 이미지 폴더 경로
    output_csv = "data/processed/steering_labels.csv"  # 저장할 CSV 파일 경로
    output_dir = "data/processed"  # 나눌 CSV 파일 저장 폴더

    # CSV 생성
    df = generate_csv_from_images(image_folder, output_csv)

    # 데이터셋 나누기
    split_dataset(df, output_dir)
