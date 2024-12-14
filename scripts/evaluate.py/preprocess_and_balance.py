import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import shutil

def crop_and_resize_images(input_folder, output_folder, crop_percentage=30, target_size=(200, 66)):
    """
    지정된 폴더 내의 모든 이미지 파일을 불러와 상단 일정 비율을 크롭하고,
    지정된 크기로 리사이즈하여 출력 폴더에 저장합니다.

    :param input_folder: 원본 이미지가 저장된 폴더 경로
    :param output_folder: 크롭 및 리사이즈된 이미지를 저장할 폴더 경로
    :param crop_percentage: 잘라낼 상단 비율 (기본값: 30)
    :param target_size: (width, height) 형태의 목표 크기 (기본값: (200, 66))
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # 기존 폴더를 비우기
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    # 지원하는 이미지 파일 확장자
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    width, height = img.size
                    crop_height = int(height * (crop_percentage / 100))
                    # 상단 30%를 잘라내고 하단 70%만 남김
                    cropped_img = img.crop((0, crop_height, width, height))
                    resized_img = cropped_img.resize(target_size, Image.ANTIALIAS)
                    resized_img.save(output_path)
                    print(f"크롭 및 리사이즈 완료: {filename}")
            except Exception as e:
                print(f"이미지 처리 중 오류 발생 ({filename}): {e}")

def get_angle_from_filename(filename):
    """
    파일명에서 각도를 추출합니다.
    예: 'capture_2024-04-27_12-00-00_angle_90.jpg'에서 90을 추출.

    :param filename: 파일명 문자열
    :return: 정수형 각도 값 또는 None
    """
    try:
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        angle_part = [part for part in parts if 'angle' in part]
        if angle_part:
            angle_str = angle_part[0].split('angle')[-1]
            return int(angle_str)
    except:
        pass
    return None

def count_images_per_angle(folder):
    """
    폴더 내 각도별 이미지 수를 세어 딕셔너리로 반환합니다.

    :param folder: 이미지가 저장된 폴더 경로
    :return: {angle: count, ...} 형태의 딕셔너리
    """
    angle_counts = {}
    for filename in os.listdir(folder):
        angle = get_angle_from_filename(filename)
        if angle is not None:
            angle_counts[angle] = angle_counts.get(angle, 0) + 1
    return angle_counts

def augment_image(image):
    """
    간단한 이미지 증강 기법을 적용하여 새로운 이미지를 생성합니다.
    여기서는 밝기 조절과 좌우 반전을 사용합니다.

    :param image: PIL.Image 객체
    :return: 증강된 PIL.Image 객체
    """
    enhancers = [
        ImageEnhance.Brightness(image),
        ImageEnhance.Contrast(image)
    ]
    augmented_images = []
    for enhancer in enhancers:
        # 밝기 증가
        augmented_images.append(enhancer.enhance(1.5))
        # 밝기 감소
        augmented_images.append(enhancer.enhance(0.7))
    
    # 좌우 반전
    augmented_images.append(image.transpose(Image.FLIP_LEFT_RIGHT))
    
    return augmented_images

def balance_dataset(input_folder, output_folder, target_count):
    """
    각도별 이미지 수를 target_count로 맞추기 위해 증강을 수행합니다.

    :param input_folder: 전처리된 이미지가 저장된 폴더 경로
    :param output_folder: 균형 맞춘 이미지를 저장할 폴더 경로
    :param target_count: 각도별 목표 이미지 수
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # 기존 폴더를 비우기
        shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    # 지원하는 이미지 파일 확장자
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    # 각도별 이미지 목록
    angle_to_images = {}
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            angle = get_angle_from_filename(filename)
            if angle is not None:
                angle_to_images.setdefault(angle, []).append(filename)

    for angle, images in angle_to_images.items():
        current_count = len(images)
        print(f"각도 {angle}도: 현재 {current_count}개, 목표 {target_count}개")
        if current_count >= target_count:
            # 목표 수 이상인 경우, 일부만 복사
            selected_images = images[:target_count]
        else:
            # 목표 수 미만인 경우, 증강을 통해 보충
            selected_images = images.copy()
            needed = target_count - current_count
            while needed > 0:
                for img_filename in images:
                    if needed <= 0:
                        break
                    img_path = os.path.join(input_folder, img_filename)
                    try:
                        with Image.open(img_path) as img:
                            augmented_imgs = augment_image(img)
                            for idx, aug_img in enumerate(augmented_imgs):
                                if needed <= 0:
                                    break
                                new_filename = f"{os.path.splitext(img_filename)[0]}_aug_{idx}.jpg"
                                aug_img.save(os.path.join(output_folder, new_filename))
                                selected_images.append(new_filename)
                                needed -= 1
                                print(f"증강 이미지 저장: {new_filename}")
                    except Exception as e:
                        print(f"증강 중 오류 발생 ({img_filename}): {e}")
        # 복사 또는 증강된 이미지를 출력 폴더로 이동
        for img_filename in selected_images:
            src_path = os.path.join(input_folder, img_filename)
            dst_path = os.path.join(output_folder, img_filename)
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"이미지 복사 중 오류 발생 ({img_filename}): {e}")

def plot_distribution(before_counts, after_counts, angles, output_path):
    """
    전처리 전후의 각도별 이미지 분포를 그래프로 시각화합니다.

    :param before_counts: 전처리 전 각도별 이미지 수 딕셔너리
    :param after_counts: 전처리 후 각도별 이미지 수 딕셔너리
    :param angles: 분석할 각도 리스트
    :param output_path: 그래프를 저장할 파일 경로
    """
    before = [before_counts.get(angle, 0) for angle in angles]
    after = [after_counts.get(angle, 0) for angle in angles]

    x = np.arange(len(angles))  # 각도 위치
    width = 0.35  # 막대 너비

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, before, width, label='Before')
    rects2 = ax.bar(x + width/2, after, width, label='After')

    # 레이블과 제목 설정
    ax.set_ylabel('Number of Images')
    ax.set_title('Image Distribution Before and After Balancing')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{angle}°" for angle in angles])
    ax.legend()

    # 값 표시
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.show()

def main():
    # 폴더 경로 설정
    original_images_folder = "images"  # Jetson Nano에서 저장한 원본 이미지 폴더
    processed_images_folder = "processed_images"  # 전처리된 이미지 저장 폴더
    balanced_images_folder = "balanced_images"  # 균형 맞춘 이미지 저장 폴더

    # 전처리: 크롭 및 리사이즈
    print("이미지 전처리 시작...")
    crop_and_resize_images(original_images_folder, processed_images_folder, crop_percentage=30, target_size=(200, 66))
    print("이미지 전처리 완료.\n")

    # 전처리 전 이미지 분포
    print("전처리 전 이미지 분포 계산...")
    before_counts = count_images_per_angle(original_images_folder)
    print("전처리 전 이미지 분포:", before_counts, "\n")

    # 전처리 후 이미지 분포
    print("전처리 후 이미지 분포 계산...")
    after_preprocess_counts = count_images_per_angle(processed_images_folder)
    print("전처리 후 이미지 분포:", after_preprocess_counts, "\n")

    # 데이터 균형 맞추기
    # 목표 수는 가장 많은 이미지 수를 기준으로 설정
    target_count = max(after_preprocess_counts.values())
    print(f"데이터 균형 맞추기 시작 (목표 수: {target_count})...")
    balance_dataset(processed_images_folder, balanced_images_folder, target_count=target_count)
    print("데이터 균형 맞추기 완료.\n")

    # 균형 맞춘 이미지 분포
    print("균형 맞춘 이미지 분포 계산...")
    after_balanced_counts = count_images_per_angle(balanced_images_folder)
    print("균형 맞춘 이미지 분포:", after_balanced_counts, "\n")

    # 시각화할 각도 리스트 (30, 60, 90, 120, 150)
    angles = [30, 60, 90, 120, 150]

    # 전처리 전과 후의 분포 시각화
    print("데이터 분포 시각화 중...")
    plot_distribution(before_counts, after_balanced_counts, angles, "image_distribution.png")
    print("데이터 분포 시각화 완료. 'image_distribution.png' 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()
