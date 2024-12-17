import os
import shutil
import re
import random
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
    overlap_ratio = 0.2
    crop_width = int(width * (1 - overlap_ratio))
    crop_height = int(height * (1 - overlap_ratio))

    for _ in range(augment_count):
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        cropped_img = image.crop((left, top, left + crop_width, top + crop_height))
        augmented_img = cropped_img.resize((width, height))
        augmentations.append(augmented_img)

    return augmentations

def extract_angle_from_filename(filename):
    """파일 이름에서 angle 값을 추출"""
    match = re.search(r"angle_(\d+)", filename)
    return int(match.group(1)) if match else None

def balance_dataset(input_folder, output_folder, target_count):
    """
    데이터셋을 언더샘플링 및 증강을 통해 균형 맞춤
    :param input_folder: 원본 데이터 폴더
    :param output_folder: 균형화된 데이터를 저장할 폴더
    :param target_count: 각도별 목표 이미지 수
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 각도별 파일 분류
    angle_to_files = {}
    for filename in os.listdir(input_folder):
        angle = extract_angle_from_filename(filename)
        if angle is not None:
            angle_to_files.setdefault(angle, []).append(filename)

    for angle, files in angle_to_files.items():
        current_count = len(files)

        # 언더샘플링: 목표 개수보다 많을 경우
        if current_count > target_count:
            print(f"각도 {angle}: 현재 {current_count}개 -> {target_count}개로 언더샘플링")
            selected_files = random.sample(files, target_count)
            for file in selected_files:
                shutil.copy(os.path.join(input_folder, file), output_folder)

        # 오버샘플링: 목표 개수보다 적을 경우
        elif current_count < target_count:
            needed_count = target_count - current_count
            print(f"각도 {angle}: 현재 {current_count}개 -> {target_count}개로 증강")
            for file in files:
                shutil.copy(os.path.join(input_folder, file), output_folder)  # 원본 복사
            while needed_count > 0:
                for file in files:
                    if needed_count <= 0:
                        break
                    img_path = os.path.join(input_folder, file)
                    with Image.open(img_path) as img:
                        augmented_imgs = augment_image_overlapping(img, 1)
                        for i, aug_img in enumerate(augmented_imgs):
                            new_filename = f"{os.path.splitext(file)[0]}_aug_{needed_count}.jpg"
                            aug_img.save(os.path.join(output_folder, new_filename))
                            needed_count -= 1

        # 정확히 맞춰졌을 경우
        else:
            print(f"각도 {angle}: 이미 목표 개수 {target_count}개 충족")
            for file in files:
                shutil.copy(os.path.join(input_folder, file), output_folder)

    print(f"모든 데이터가 '{output_folder}'에 균형 맞춰졌습니다.")

def plot_distribution(before_counts, after_counts, angles, output_path):
    """전처리 전후의 각도별 이미지 분포를 그래프로 시각화"""
    before = [before_counts.get(angle, 0) for angle in angles]
    after = [after_counts.get(angle, 0) for angle in angles]

    x = np.arange(len(angles))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before, width, label='Before', alpha=0.7)
    rects2 = ax.bar(x + width/2, after, width, label='After', alpha=0.7)

    ax.set_ylabel('Number of Images')
    ax.set_title('Image Distribution Before and After Balancing')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{angle}°" for angle in angles])
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    input_folder = r"C:\Users\USER\Desktop\programing\code\data\processed\cropped_resized"
    output_folder = r"C:\Users\USER\Desktop\programing\code\data\processed\balanced"
    target_count = 500

    # 전처리 전 이미지 분포 계산
    angle_counts_before = {}
    for filename in os.listdir(input_folder):
        angle = extract_angle_from_filename(filename)
        if angle is not None:
            angle_counts_before[angle] = angle_counts_before.get(angle, 0) + 1

    # 데이터 증강 및 언더샘플링
    balance_dataset(input_folder, output_folder, target_count)

    # 전처리 후 이미지 분포 계산
    angle_counts_after = {}
    for filename in os.listdir(output_folder):
        angle = extract_angle_from_filename(filename)
        if angle is not None:
            angle_counts_after[angle] = angle_counts_after.get(angle, 0) + 1

    # Before와 After의 모든 각도 가져오기
    angles = sorted(set(angle_counts_before.keys()) | set(angle_counts_after.keys()))

    # 그래프 출력
    plot_distribution(angle_counts_before, angle_counts_after, angles,
                      r"C:\Users\USER\Desktop\programing\code\data\processed\balanced\image_distribution.png")
