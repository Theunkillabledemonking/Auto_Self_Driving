import cv2
import numpy as np
import os

def load_calibration_parameters(calibration_file):
    # 캘리브레이션 파라미터 로드
    with np.load(calibration_file) as data:
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
    return camera_matrix, dist_coeffs

def undistort_image(img, camera_matrix, dist_coeffs):
    # 이미지 크기 얻기
    h, w = img.shape[:2]
    # 새로운 카메라 매트릭스 계산
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))
    # 이미지 보정
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
    # 필요에 따라 이미지를 크롭
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def process_image_with_undistortion(image_path, camera_matrix, dist_coeffs):
    img = cv2.imread(image_path)
    undistorted_img = undistort_image(img, camera_matrix, dist_coeffs)
    # 이후 이미지 처리 수행 (예: 차선 인식)
    processed_img = process_image(undistorted_img)
    return processed_img

def process_image(img):
    # 기존에 작성한 이미지 처리 함수
    # ...
    return img  # 처리된 이미지 반환

def main():
    # 캘리브레이션 파라미터 로드
    camera_matrix, dist_coeffs = load_calibration_parameters('calibration_data.npz')
    
    # 이미지 폴더 설정
    folder = 'path_to_your_images'  # 이미지가 저장된 폴더 경로
    output_folder = 'output_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 이미지 처리
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder, filename)
            processed_img = process_image_with_undistortion(image_path, camera_matrix, dist_coeffs)
            output_path = os.path.join(output_folder, 'processed_' + filename)
            cv2.imwrite(output_path, processed_img)
            print(f"처리된 이미지 저장: {output_path}")

if __name__ == '__main__':
    main()
