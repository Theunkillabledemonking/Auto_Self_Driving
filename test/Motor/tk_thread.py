import threading
import Jetson.GPIO as GPIO
import cv2
import time
import datetime
import tkinter as tk

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# 핀 번호 설정 (필요에 따라 수정하세요)
servo_pin = 33  # 서보 모터 제어 핀
dc_motor_pwm_pin = 32  # DC 모터 PWM 제어 핀
dc_motor_dir_pin1 = 29  # DC 모터 방향 제어 핀 1
dc_motor_dir_pin2 = 31  # DC 모터 방향 제어 핀 2

# GPIO 핀 설정
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_pin, 50)  # 서보 모터용 50Hz
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DC 모터용 1kHz
servo.start(0)
dc_motor_pwm.start(0)

# 현재 속도 및 각도 변수
current_speed = 0
current_angle = 90

# 키 상태를 추적하기 위한 집합
keys_pressed = set()

# 스레드 종료를 위한 이벤트
stop_event = threading.Event()

# 서보모터와 DC 모터를 하나의 메서드로 관리하는 함수
def control_motors(angle=None, speed=None, direction="forward"):
    if angle is not None:
        duty_cycle = 2 + (angle / 18)
        if duty_cycle < 2 or duty_cycle > 12:
            print(f"Invalid duty cycle: {duty_cycle}")
        else:
            try:
                servo.ChangeDutyCycle(duty_cycle)
                time.sleep(0.1)
                servo.ChangeDutyCycle(0)
            except Exception as e:
                print(f"Error setting servo angle: {e}")

    if speed is not None:
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        try:
            dc_motor_pwm.ChangeDutyCycle(speed)
        except Exception as e:
            print(f"Error setting DC motor speed: {e}")

# 키 입력에 따라 모터를 제어하는 함수
def update_motors():
    global current_speed, current_angle
    # 기본 값은 변경하지 않음
    speed = current_speed
    angle = current_angle
    direction = "forward"

    if 'w' in keys_pressed:
        speed = min(current_speed + 5, 75)
    if 's' in keys_pressed:
        speed = max(current_speed - 5, 0)

    if 'a' in keys_pressed:
        angle = max(0, current_angle - 5)
    if 'd' in keys_pressed:
        angle = min(180, current_angle + 5)

    # 모터 제어 함수 호출
    control_motors(angle=angle, speed=speed, direction=direction)

    # 현재 상태 업데이트
    current_speed = speed
    current_angle = angle

    print(f"속도: {current_speed}%, 조향 각도: {current_angle}도")

# 스레드 집합 안정성
keys_lock = threading.Lock()

# 키가 눌렸을 때 호출되는 함수
def on_key_press(event):
    key = event.keysym.lower()
    if key == 'q':
        print("종료 키 입력됨")
        stop_event.set()
        root.quit()
    else:
        with keys_lock:
            keys_pressed.add(key)
        update_motors()

# 키가 떼어졌을 때 호출되는 함수
def on_key_release(event):
    key = event.keysym.lower()
    with keys_lock:
        if key in keys_pressed:
            keys_pressed.remove(key)
    update_motors()

# 웹캠에서 실시간 영상을 저장하고 파일로 저장하는 함수
def record_video():
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera")
            return

        # 카메라 설정 (필요에 따라 조정)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"/home/haru/sg/My_self_driving/OpenCV/{timestamp}.avi"  # 원하는 경로로 변경 가능
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

        while not stop_event.is_set():
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                print("Failed to read frame from camera")
                break

        cap.release()
        out.release()
        print(f"Video saved as {video_filename}")
    except Exception as e:
        print(f"Error in record_video: {e}")

# 메인 실행 함수
def main():
    global root
    try:
        control_motors(angle=current_angle)
        print("W/S로 속도 조절, A/D로 조향합니다. 동시에 눌러 방향 전환이 가능합니다. 종료하려면 'Q'를 누르세요.")

        # 웹캠 촬영 및 저장을 별도 스레드에서 병렬 실행
        video_thread = threading.Thread(target=record_video)
        video_thread.start()

        # RC 자동차 제어를 메인 스레드에서 실행 (키보드 입력 처리)
        root = tk.Tk()
        root.title("RC 자동차 제어")
        root.bind('<KeyPress>', on_key_press)
        root.bind('<KeyRelease>', on_key_release)
        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

        # 비디오 스레드 종료 대기
        video_thread.join()
    except Exception as e:
        print(f"Unexpected error in main: {e}")
    finally:
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()
        print("GPIO 정리 및 프로그램 종료")

def on_closing():
    stop_event.set()
    root.quit()

if __name__ == "__main__":
    main()
