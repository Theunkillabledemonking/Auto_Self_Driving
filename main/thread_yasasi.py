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

    # 자동 녹화 시작
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"recording_{timestamp}.avi"
    out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
    print(f"녹화를 시작합니다: {filename}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("영상을 읽을 수 없습니다.")
            break

        cv2.imshow('Live Feed', frame)

        # 녹화 중인 프레임 저장
        out.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 종료 키
            print("운행 종료. 녹화를 중지합니다.")
            break

    cap.release()
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
        duty_cycle = 2 + (angle / 18)
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)
        servo.ChangeDutyCycle(0)

    try:
        current_speed = 0
        max_speed = 50
        booster_speed = 100
        current_servo_angle = 90

        print("W, S로 전진/후진을 조작하세요. A, D로 좌회전/우회전. Shift로 부스터를 사용합니다. 'q'로 종료합니다.")
        while True:
            if keyboard.is_pressed('w') and keyboard.is_pressed('a'):  # 좌회전 + 전진
                current_speed = min(current_speed + 2, max_speed)
                current_servo_angle = max(45, current_servo_angle - 5)  # 서보를 좌측으로 회전
                set_dc_motor(current_speed, "forward")
                set_servo_angle(current_servo_angle)
                print(f"좌회전 전진 중: 속도 {current_speed}%, 각도 {current_servo_angle}도")
            elif keyboard.is_pressed('w') and keyboard.is_pressed('d'):  # 우회전 + 전진
                current_speed = min(current_speed + 2, max_speed)
                current_servo_angle = min(135, current_servo_angle + 5)  # 서보를 우측으로 회전
                set_dc_motor(current_speed, "forward")
                set_servo_angle(current_servo_angle)
                print(f"우회전 전진 중: 속도 {current_speed}%, 각도 {current_servo_angle}도")
            elif keyboard.is_pressed('w'):  # 전진 (서서히 속도 증가)
                current_speed = min(current_speed + 2, max_speed)
                set_dc_motor(current_speed, "forward")
                print(f"전진 중: 속도 {current_speed}%")
            elif keyboard.is_pressed('s'):  # 후진
                current_speed = 30  # 고정 속도로 후진
                set_dc_motor(current_speed, "backward")
                print(f"후진 중: 속도 {current_speed}%")
            elif keyboard.is_pressed('shift'):  # 부스터
                set_dc_motor(booster_speed, "forward")
                print(f"부스터 사용: 속도 {booster_speed}%")
                time.sleep(0.5)  # 부스터 지속 시간
                current_speed = max_speed
            elif keyboard.is_pressed('a'):  # 좌회전
                current_servo_angle = max(45, current_servo_angle - 5)
                set_servo_angle(current_servo_angle)
                print(f"좌회전 중: 각도 {current_servo_angle}도")
            elif keyboard.is_pressed('d'):  # 우회전
                current_servo_angle = min(135, current_servo_angle + 5)
                set_servo_angle(current_servo_angle)
                print(f"우회전 중: 각도 {current_servo_angle}도")
            else:  # 서서히 정지
                if current_speed > 0:
                    current_speed = max(current_speed - 1, 0)
                    set_dc_motor(current_speed, "forward")
                    print(f"서서히 감속 중: 속도 {current_speed}%")
                else:
                    set_dc_motor(0, "stop")
            
            if keyboard.is_pressed('q'):  # 종료
                print("운행 종료.")
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
