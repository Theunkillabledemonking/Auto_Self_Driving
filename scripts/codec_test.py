import cv2


# 웹캠 연결
cap = cv2.VideoCapture(0)

# 코덱 설정 및 VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'XVID') # 또는 *'MJPG'
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 프레임 저장
        out.write(frame)
        
        # 프레임 화면에 표시
        cv2.imshow('frame', frame)
        
        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else:
        break
    
# 자원 해제
cap.release()
cap.release()
cv2.destoryAllWindows()