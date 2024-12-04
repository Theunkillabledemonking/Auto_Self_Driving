import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    filenames = []
    angles = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
                filenames.append(filename)
                # 파일명에서 각도 추출
                angle = extract_angle_from_filename(filename)
                angles.append(angle)
    return images, filenames, angles

def extract_angle_from_filename(filename):
    # 파일명에서 'angle_anglevalue' 부분 추출
    # 예시 파일명: image_timestamp_speed_60_angle_90.jpg
    try:
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part == 'angle':
                angle_value = float(parts[i + 1].split('.')[0])  # '.jpg' 제거
                return angle_value
    except (IndexError, ValueError):
        pass
    return None  # 각도 정보를 추출하지 못한 경우

def gaussian_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def color_filter(img):
    # 이미지 색상 공간을 BGR에서 HSV로 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 흰색 범위 정의 (HSV에서)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 25, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    # 원본 이미지에 마스크 적용
    filtered = cv2.bitwise_and(img, img, mask=mask_white)
    return filtered

def canny_edge(img, low_threshold=50, high_threshold=150):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, angle):
    height = img.shape[0]
    width = img.shape[1]
    # 각도에 따라 관심 영역 조정 (필요에 따라 조정)
    # 여기서는 각도를 사용하여 ROI를 동적으로 조정하는 예시입니다.
    if angle is not None:
        # 각도가 큰 경우(좌우로 많이 꺾인 경우) ROI를 넓게 설정
        offset = int((abs(angle - 90) / 90) * width * 0.2)
    else:
        offset = 0

    # 관심 영역을 나타내는 다각형 정의
    polygons = np.array([
        [
            (offset, height),
            (width - offset, height),
            (width // 2, int(height * 0.6))
        ]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img):
    # 허프 변환 파라미터 설정
    rho = 2             # 거리 해상도 (픽셀)
    theta = np.pi / 180 # 각도 해상도 (라디안)
    threshold = 50      # 누적값 임계값
    min_line_length = 40  # 최소 선 길이
    max_line_gap = 100    # 최대 선 간격
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 선분을 이미지에 그림
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return line_image

def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    if lines is None:
        return None, None
    for line in lines:
        for x1, y1, x2, y2 in line:
            # 선분의 기울기와 절편 계산
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_line = make_points(img, np.mean(left_fit, axis=0)) if left_fit else None
    right_line = make_points(img, np.mean(right_fit, axis=0)) if right_fit else None
    return left_line, right_line

def make_points(img, line):
    height, width, _ = img.shape
    slope, intercept = line
    y1 = height  # 이미지의 아래쪽
    y2 = int(y1 * 0.6)  # 이미지의 중간 부분
    if slope == 0:
        slope = 0.1  # 0으로 나누는 오류 방지
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def combine_images(img1, img2, α=0.8, β=1, λ=0):
    return cv2.addWeighted(img1, α, img2, β, λ)

def process_image(image, angle):
    # 가우시안 블러 적용
    blurred = gaussian_blur(image)
    # 색상 필터 적용
    filtered = color_filter(blurred)
    # 그레이스케일 변환
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    # 캐니 에지 검출
    edges = canny_edge(gray)
    # 관심 영역 설정 (각도에 따라 조정)
    cropped_edges = region_of_interest(edges, angle)
    # 허프 변환을 통한 선 검출
    lines = hough_lines(cropped_edges)
    # 선분을 평균화하여 차선 결정
    left_line, right_line = average_slope_intercept(image, lines)
    line_image = np.zeros_like(image)
    if left_line is not None:
        x1, y1, x2, y2 = left_line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    if right_line is not None:
        x1, y1, x2, y2 = right_line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    # 원본 이미지와 선 이미지를 결합
    combined = combine_images(image, line_image)
    return combined

def main():
    folder = 'path_to_your_images'  # 여기에 이미지 폴더 경로를 입력하세요
    images, filenames, angles = load_images_from_folder(folder)
    output_folder = 'output_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for idx, image in enumerate(images):
        angle = angles[idx]
        result = process_image(image, angle)
        # 결과 이미지를 저장할 때 각도 정보를 포함
        output_filename = 'processed_angle_{0}_{1}'.format(angle, filenames[idx])
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, result)
        # 결과를 화면에 표시하려면 아래 주석을 해제하세요
        # cv2.imshow('Result', result)
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
