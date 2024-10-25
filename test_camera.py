import cv2

# 카메라 초기화
cap = cv2.VideoCapture(0)

# 비디오 저장을 위한 설정
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if ret:
        # 영상 출력
        cv2.imshow('Camera Feed', frame)

        # 영상 저장
        out.write(frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()