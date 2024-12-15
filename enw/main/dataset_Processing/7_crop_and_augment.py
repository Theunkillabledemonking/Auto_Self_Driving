import cv2
import os

# 원본 이미지 폴더
input_folder = "C:/Users/USER/Desktop/programing/code/data/frames"

# 증강 이미지 저장 폴더
output_folder = "C:/Users/USER/Desktop/programing/code/data/augmented_frames"
os.makedirs(output_folder, exist_ok=True)

# 폴더 내 모든 파일 처리
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):  # 다양한 이미지 파일 처리
        # 원본 이미지 경로
        image_path = os.path.join(input_folder, filename)

        # 이미지 불러오기
        image = cv2.imread(image_path)
        if image is None:
            print(f"이미지를 불러오지 못했습니다: {image_path}")
            continue

        # 이미지 크기 확인
        height, width, _ = image.shape
        print(f"원본 이미지 크기: {width}x{height}")

        # ROI 설정 (상단 30% 제거)
        roi_top = int(height * 0.3)
        roi = image[roi_top:height, :]
        print(f"ROI 크기: {roi.shape[1]}x{roi.shape[0]}")

        # 크기 조정 (모델 입력 크기로)
        roi_resized = cv2.resize(roi, (200, 66))
        print(f"리사이즈 후 크기: {roi_resized.shape[1]}x{roi_resized.shape[0]}")

        # 결과 이미지 저장 경로
        output_path = os.path.join(output_folder, f"cropped_{os.path.splitext(filename)[0]}.jpg")

        # 결과 저장
        cv2.imwrite(output_path, roi_resized)
        print(f"전처리된 이미지를 저장했습니다: {output_path}")

print("모든 이미지의 전처리가 완료되었습니다.")
