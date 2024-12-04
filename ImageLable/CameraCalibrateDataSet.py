import cv2
import numpy as np
import glob

# 체스보드 패턴의 내부 코너 수 설정 (가로, 세로)
corner_rows = 6
corner_cols = 9

# 체스보드 패턴의 코너 좌표 준비
objp = np.zeros((corner_rows * corner_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_cols, 0:corner_rows].T.reshape(-1, 2)

# 3D 점과 2D 이미지 점을 저장할 배열
objpoints = []  # 실제 세계의 3D 점
imgpoints = []  # 이미지 평면의 2D 점

# 캘리브레이션 이미지 로드
images = glob.glob('calibration_images/*.jpg')  # 캘리브레이션 이미지 경로

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, (corner_cols, corner_rows), None)

    if ret == True:
        objpoints.append(objp)

        # 코너 위치를 더 정확하게 수정
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria=None)
        imgpoints.append(corners2)

        # 코너 그리기 (옵션)
        cv2.drawChessboardCorners(img, (corner_cols, corner_rows), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# 캘리브레이션 결과 저장
np.savez('calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
