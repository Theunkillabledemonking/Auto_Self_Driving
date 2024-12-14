import os
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def augment_image_overlapping(image, augment_count):
    """
    이미지 증강: Overlapping을 이용한 부분 자르기 및 생성

    :param image: PIL.Image 객체
    :param augment_count: 생성할 증강 이미지 수
    :return: 증강된 이미지 리스트
    """
    augmentations = []
    width, height = image.size

    # Overlapping 비율 및 자르기
    overlap_ratio = 0.2  # 20% Overlapping
    crop_width = int(width * (1 - overlap_ratio))
    crop_height = int(height * (1 - overlap_ratio))

    for i in range(augment_count):
        left = np.random.randint(0, width - crop_width + 1)
        top = np.random.randint(0, height - crop_height + 1)
        cropped_img = image.crop((left, top, left + crop_width, top + crop_height))

        # 자른 이미지를 원래 크기로 리사이즈
        augmented_img = cropped_img.resize((width, height))
        augmentations.append(augmented_img)

    return augmentations

def balance_dataset(input_folder, output_folder, target_count):
    """
    Overlapping 방식으로 각도별 데이터 증강하여 균형 맞춤

    :param input_folder: 원본 데이터 폴더
    :param output_folder: 증강된 데이터를 저장할 폴더
    :param target_count: 각도별 목표 이미지 수
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 각도별 이미지 수 세기
    angle_to_files = {}
    for filename in os.listdir(input_folder):
        if "angle_" in filename:
            angle = int(filename.split("angle_")[1].split(".")[0])
            angle_to_files.setdefault(angle, []).append(filename)

    for angle, files in angle_to_files.items():
        current_count = len(files)
        if current_count >= target_count:
            print(f"각도 {angle}: 현재 {current_count}개로 충분함")
            for file in files:
                shutil.copy(os.path.join(input_folder, file), output_folder)
        else:
            needed_count = target_count - current_count
            print(f"각도 {angle}: 현재 {current_count}개, {needed_count}개 더 필요")
            
            # 증강 실행
            for file in files:
                img_path = os.path.join(input_folder, file)
                with Image.open(img_path) as img:
                    augmentations = augment_image_overlapping(img, needed_count)
                    for i, augmented_img in enumerate(augmentations):
                        new_filename = f"{os.path.splitext(file)[0]}_aug_{i}.jpg"
                        augmented_img.save(os.path.join(output_folder, new_filename))
                        needed_count -= 1
                        if needed_count <= 0:
                            break
                if needed_count <= 0:
                    break
            print(f"각도 {angle} 데이터 증강 완료")

    print(f"모든 데이터가 '{output_folder}'로 균형 맞춰졌습니다.")

def plot_distribution(before_counts, after_counts, angles, output_path):
    """
    전처리 전후의 각도별 이미지 분포를 그래프로 시각화합니다.
    """
    before = [before_counts.get(angle, 0) for angle in angles]
    after = [after_counts.get(angle, 0) for angle in angles]

    x = np.arange(len(angles))  # 각도 위치
    width = 0.35  # 막대 너비

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before, width, label='Before', alpha=0.7)
    rects2 = ax.bar(x + width/2, after, width, label='After', alpha=0.7)

    # 레이블 설정
    ax.set_ylabel('Number of Images')
    ax.set_title('Image Distribution Before and After Balancing')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{angle}°" for angle in angles])
    ax.legend()

    # 그래프 위에 값 표시
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 약간 위로 이동
                        textcoords="offset points",
                        ha='center', va='bottom')

    fig.tight_layout()
    plt.savefig(output_path)
    plt.show()

# 실행 예시
if __name__ == "__main__":
    input_folder = r"C:\Users\USER\Desktop\programing\code\data\processed\resized"  # 리사이즈된 데이터
    output_folder = r"C:\Users\USER\Desktop\programing\code\data\balanced"  # 균형화된 데이터 저장
    target_count = 500  # 각도별 목표 이미지 수

    # 전처리 전 이미지 분포 계산
    angle_counts_before = {}
    for filename in os.listdir(input_folder):
        if "angle_" in filename:
            angle = int(filename.split("angle_")[1].split(".")[0])
            angle_counts_before[angle] = angle_counts_before.get(angle, 0) + 1

    # 데이터 증강 및 균형화
    balance_dataset(input_folder, output_folder, target_count)

    # 전처리 후 이미지 분포 계산
    angle_counts_after = {}
    for filename in os.listdir(output_folder):
        if "angle_" in filename:
            angle = int(filename.split("angle_")[1].split(".")[0])
            angle_counts_after[angle] = angle_counts_after.get(angle, 0) + 1

    # 그래프 분포도 시각화
    angles = sorted(angle_counts_before.keys())
    plot_distribution(angle_counts_before, angle_counts_after, angles, "image_distribution.png")
