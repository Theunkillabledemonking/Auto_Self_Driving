import os
from PIL import Image

def resize_images(input_folder, output_folder, target_size=(200, 66)):
    """
    지정된 폴더 내의 모든 이미지 파일을 불러와 지정된 크기로 리사이즈하고,
    리사이즈된 이미지를 출력 폴더에 저장합니다.

    :param input_folder: 원본 이미지가 저장된 폴더 경로
    :param output_folder: 리사이즈된 이미지를 저장할 폴더 경로
    :param target_size: (width, height) 형태의 목표 크기 (기본값: (200, 66))
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 지원하는 이미지 파일 확장자
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    # 리사이즈: LANCZOS 필터 사용
                    resized_img = img.resize(target_size, Image.LANCZOS)
                    resized_img.save(output_path)
                    print(f"이미지 리사이즈 완료: {filename}")
            except Exception as e:
                print(f"이미지 리사이즈 중 오류 발생 ({filename}): {e}")

if __name__ == "__main__":
    # 크롭된 이미지가 저장된 폴더 경로
    input_folder = "data/processed/cropped"  # crop_images.py에서 생성된 폴더

    # 리사이즈된 이미지를 저장할 폴더 경로
    output_folder = "data/processed/resized"

    # 목표 크기 (width, height)
    target_size = (200, 66)

    resize_images(input_folder, output_folder, target_size)
    print("모든 이미지 리사이즈 완료.")
