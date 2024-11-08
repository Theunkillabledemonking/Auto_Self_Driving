# -*- coding: utf-8 -*-

import cv2
import datetime

def main():
    # 웹캠 연결 (디바이스는 인덱스는 보통 0입니다.)
    cap = cv2.VideoCapture(0)
    
    # 웹캠이 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    # 한 프레임 읽기
    ret, frame = cap.read()
    
    # 프레임이 정상적으로 읽혔는지 확인
    if ret:
        # 현재 시간을 기반으로 파일 이름 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_image_{timestamp}.jpg"
        
        # 사진 저장
        cv2.imwrite(filename, frame)
        print(f"사진이 '{filename}' 으로 저장되었습니다.")
        
        # 사진을 화면에 출력
        cv2.imshow("Captured Image", frame)
        cv2.waitKey(0)
    else:
        print("프레임을 가져올 수 없습니다.")
        
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()