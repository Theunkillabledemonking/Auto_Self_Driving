import os
from PIL import Image

def crop_and_resize_images(input_folder, output_folder, crop_percentage=40, target_size=(200, 66)):
    """
    지정된 폴더 내의 모든 이미지 파일을 불러와 상단 일정 비율을 크롭하고,
    지정된 크기로 리사이즈하여 출력 폴더에 저장합니다.

    :param input_folder: 원본 이미지가 저장된 폴더 경로
    :param output_folder: 크롭 및 리사이즈된 이미지를 저장할 폴더 경로
    :param crop_percentage: 잘라낼 상단 비율 (기본값: 45)
    :param target_size: (width, height) 형태의 목표 크기 (기본값: (200, 66))
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 지원하는 이미지 파일 확장자
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_extensions):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                with Image.open(input_path) as img:
                    width, height = img.size
                    crop_height = int(height * (crop_percentage / 100))
                    # 상단 crop_percentage%를 잘라내고 나머지 사용
                    cropped_img = img.crop((0, crop_height, width, height))
                    resized_img = cropped_img.resize(target_size, Image.ANTIALIAS)
                    resized_img.save(output_path)
                    print(f"크롭 및 리사이즈 완료: {filename}")
            except Exception as e:
                print(f"이미지 처리 중 오류 발생 ({filename}): {e}")

# 실행 예제
if __name__ == "__main__":
    input_folder = r"C:\Users\USER\Desktop\programing\code\data\images"  # 원본 이미지 폴더
    output_folder = r"C:\Users\USER\Desktop\programing\code\data\processed\cropped_resized"  # 결과 저장 폴더
    crop_and_resize_images(input_folder, output_folder, crop_percentage=45, target_size=(200, 66))
