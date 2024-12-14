import os
from PIL import Image

def crop_top_percentage(input_folder, output_folder, crop_percentage=30):
    """
    지정된 폴더 내의 모든 이미지 파일을 불러와 상단 일정 비율을 잘라내고,
    잘라낸 이미지를 출력 폴더에 저장합니다.

    :param input_folder: 원본 이미지가 저장된 폴더 경로
    :param output_folder: 잘라낸 이미지를 저장할 폴더 경로
    :param crop_percentage: 잘라낼 상단 비율 (기본값: 30)
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
                    width, height = img.size
                    crop_height = int(height * (crop_percentage / 100))
                    # 상단 30%를 잘라내고 하단 70%만 남김
                    cropped_img = img.crop((0, crop_height, width, height))
                    cropped_img.save(output_path)
                    print(f"이미지 처리 완료: {filename}")
            except Exception as e:
                print(f"이미지 처리 중 오류 발생 ({filename}): {e}")

if __name__ == "__main__":
    # 원본 이미지가 저장된 폴더 경로
    input_folder = "images"  # 예: Jetson에서 'images' 폴더에 저장된 경우

    # 잘라낸 이미지를 저장할 폴더 경로
    output_folder = "cropped_images"

    # 상단 30%를 잘라냄
    crop_percentage = 30

    crop_top_percentage(input_folder, output_folder, crop_percentage)
    print("모든 이미지 처리 완료.")
