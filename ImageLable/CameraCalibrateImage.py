import cv2
import numpy as np

# 캘리브레이션 데이터 로드
with np.load('calibration_data.npz') as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

# 이미지 로드
img = cv2.imread('path_to_your_image.jpg')
h, w = img.shape[:2]

# 새로운 카메라 매트릭스 계산 (전체 이미지를 보정하도록 alpha=1 설정)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))

# 이미지 보정
dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# 필요에 따라 이미지를 크롭
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

# 보정된 이미지로 이미지 처리 수행
# 예: process_image(dst)
