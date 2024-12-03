import threading
import cv2
import datetime
import Jetson.GPIO as GPIO
import time
import keyboard

# 카메라 영상 처리 및 녹화 함수
def camera_processing():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    recording = False
    out = None

    print("카메라 활성화 완료. 'r' 키를 눌러 녹화를 시작/중지하고, 'q' 키로 프로그램을 종료합니다.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상을 읽을 수 없습니다.")
            break

        cv2.imshow('Live Feed', frame)

        # 녹화 상태에서 프레임 저장
        if recording and out is not None:
            out.write(frame)

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # 녹화 시작/중지
            if not recording:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f"recording_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
                recording = True
                print(f"녹화를 시작합니다: {filename}")
            else:
                recording = False
                out.release()
                out = None
                print("녹화를 중지합니다.")

        if key == ord('q'):  # 종료 키
            print("운행 종료.")
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

# 모터 제어 함수
def motor_control():
    servo_pin = 33
    dc_motor_pwm_pin = 32
    dc_motor_dir_pin1 = 29
    dc_motor_dir_pin2 = 31

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(servo_pin, GPIO.OUT)
    GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
    GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
    GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

    servo = GPIO.PWM(servo_pin, 50)
    dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)
    servo.start(0)
    dc_motor_pwm.start(0)

    def set_dc_motor(speed, direction):
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        elif direction == "stop":
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        dc_motor_pwm.ChangeDutyCycle(speed)

    def set_servo_angle(angle):
        # 서보 각도 설정 함수
        duty_cycle = 2 + (angle / 18)
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)
        servo.ChangeDutyCycle(0)

    try:
        current_speed = 60  # 기본 속도
        current_servo_angle = 90  # 중립 각도 (90도)
        booster_speed = 80

        print("W: 전진 | S: 후진 | A: 좌회전 | D: 우회전 | Shift: 부스터 | Q: 종료")
        while True:
            if keyboard.is_pressed('w') and keyboard.is_pressed('a'):  # 좌회전 + 전진
                current_servo_angle = max(80, current_servo_angle - 2)  # 작게 좌회전
                set_servo_angle(current_servo_angle)
                set_dc_motor(current_speed, "forward")
                print(f"좌회전 전진: 각도 {current_servo_angle}도")
            elif keyboard.is_pressed('w') and keyboard.is_pressed('d'):  # 우회전 + 전진
                current_servo_angle = min(100, current_servo_angle + 2)  # 작게 우회전
                set_servo_angle(current_servo_angle)
                set_dc_motor(current_speed, "forward")
                print(f"우회전 전진: 각도 {current_servo_angle}도")
            elif keyboard.is_pressed('w'):  # 직진
                set_servo_angle(90)  # 각도 중립
                set_dc_motor(current_speed, "forward")
                print(f"직진: 속도 {current_speed}%")
            elif keyboard.is_pressed('s'):  # 후진
                set_servo_angle(90)  # 각도 중립
                set_dc_motor(55, "backward")
                print("후진 중: 속도 55%")
            elif keyboard.is_pressed('shift'):  # 부스터
                set_dc_motor(booster_speed, "forward")
                print(f"부스터 사용: 속도 {booster_speed}%")
            elif keyboard.is_pressed('a'):  # 작은 좌회전
                current_servo_angle = max(80, current_servo_angle - 2)  # 좌회전 각도 제한
                set_servo_angle(current_servo_angle)
                print(f"좌회전 중: 각도 {current_servo_angle}도")
            elif keyboard.is_pressed('d'):  # 작은 우회전
                current_servo_angle = min(100, current_servo_angle + 2)  # 우회전 각도 제한
                set_servo_angle(current_servo_angle)
                print(f"우회전 중: 각도 {current_servo_angle}도")
            else:  # 정지 또는 감속
                set_dc_motor(0, "stop")
                print("정지 또는 감속 중")

            if keyboard.is_pressed('q'):  # 종료
                print("운행 종료")
                break

            time.sleep(0.1)
    finally:
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()

# 메인 함수 정의
def main():
    camera_thread = threading.Thread(target=camera_processing)
    motor_thread = threading.Thread(target=motor_control)

    camera_thread.start()
    motor_thread.start()

    camera_thread.join()
    motor_thread.join()

# 프로그램 진입점
if __name__ == "__main__":
    main()
