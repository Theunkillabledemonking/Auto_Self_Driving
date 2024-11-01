import cv2

def main():
    # 웹캠 연결 (디바이스 인덱스는 보통 0입니다)
    cap = cv2.VideoCapture(0)
    
    # 웹캠이 정상적으로 열렸는지 확인
    if not cap.isOpend():
        print("웹캠을 열 수 없습니다.")
        return
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        # 프레임이 정상적으로 읽혔는지 확인
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        # 프레임 출력
        cv2.imshow('Webcam Video', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # 자원 해제
        cap.realse()
        cv2.destoryAllwindows()
        
if __name__ == "__main__":
    main()
        
        