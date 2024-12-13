import cv2

def main():
    # 웹캠 연결 (디바이스 인덱스는 보통 0입니다.)
    cap = cv2.VideoCapture(0)
    
    # 웹캠이 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return
    
    # 영상 저장을 위한 설정 (코덱, 파일명, FPS, 프레임 크기)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = 20 # 초당 프레임 수
    output_filename = 'output_video.avi'
    fourcc = cv2 .VideoWriter_fourcc(*'XVID') # 코덱 설정 (XVID 사용)
    
    # VideoWriter 객체 생성
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        
        # 프레임이 정상적으로 읽혔는지 확인
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break
        
        # 프레임 저장
        out.write(frame)
        
        # 프레임 출력 (영상 저장 중에도 화면에 출력 가능)
        cv2.imshow('Recoding Video', frame)
        
        # 'q'키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
