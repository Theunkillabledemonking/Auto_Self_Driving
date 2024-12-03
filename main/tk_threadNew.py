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
servo.start(7.5)  # 중립 위치 (7.5%)
dc_motor_pwm.start(0)  # 초기 속도 0%

# 현재 속도 및 각도 변수
current_speed = 0  # 초기 속도 0%
current_angle = 90  # 중립 각도

# 키 상태를 추적하기 위한 집합
keys_pressed = set()

# 스레드 종료를 위한 이벤트
stop_event = threading.Event()
recording_event = threading.Event()

# 서보모터와 DC 모터를 하나의 메서드로 관리하는 함수
def control_motors(angle=None, speed=None, direction=None):
    if angle is not None:
        duty_cycle = 2 + (angle / 18)
        if duty_cycle < 2 or duty_cycle > 12:
            print(f"Invalid duty cycle: {duty_cycle}")
        else:
            try:
                servo.ChangeDutyCycle(duty_cycle)
                # time.sleep(0.1)
                # servo.ChangeDutyCycle(0)  # 이 부분을 제거하여 서보모터가 지속적으로 각도를 유지하도록 함
            except Exception as e:
                print(f"Error setting servo angle: {e}")

    if speed is not None and direction is not None:
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        else:  # 정지 상태
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        try:
            dc_motor_pwm.ChangeDutyCycle(speed)
        except Exception as e:
            print(f"Error setting DC motor speed: {e}")

# 키 입력에 따라 모터를 제어하는 함수
def update_motors():
    global current_speed, current_angle
    speed = 0
    direction = None  # 정지 상태

    # 기본 값은 변경하지 않음
    angle = current_angle

    # 각도 제한 및 조정 값 설정
    min_angle = 70  # 최소 조향 각도 (왼쪽 최대)
    max_angle = 110  # 최대 조향 각도 (오른쪽 최대)
    angle_step = 2  # 각도 변경 단위

    with keys_lock:
        if 'a' in keys_pressed:
            angle = max(min_angle, current_angle - angle_step)
        if 'd' in keys_pressed:
            angle = min(max_angle, current_angle + angle_step)
        if 'w' in keys_pressed:
            speed = 55  # 원하는 속도로 설정
            direction = "forward"
        elif 's' in keys_pressed:
            speed = 55  # 동일한 속도로 후진
            direction = "backward"
        else:
            speed = 0
            direction = None  # 모터 정지

    # 모터 제어 함수 호출
    control_motors(angle=angle, speed=speed, direction=direction)

    # 현재 상태 업데이트
    current_speed = speed
    current_angle = angle

    if direction:
        print(f"속도: {current_speed}%, 방향: {direction}, 조향 각도: {current_angle}도")
    else:
        print(f"모터 정지, 조향 각도: {current_angle}도")

# 스레드 집합 안정성
keys_lock = threading.Lock()

# 키가 눌렸을 때 호출되는 함수
def on_key_press(event):
    key = event.keysym.lower()
    if key == 'q':
        print("종료 키 입력됨")
        stop_event.set()
        recording_event.set()
        root.quit()
    elif key == 'r':
        print("녹화 시작 키 입력됨")
        recording_event.set()
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
        # 녹화 시작 신호를 기다림
        recording_event.wait()
        if stop_event.is_set():
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera")
            return

        # 카메라 설정 (필요에 따라 조정)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 20)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"/home/sg/sg/My_self_driving/Video/{timestamp}.avi"  # 원하는 경로로 변경 가능
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))

        print("녹화를 시작합니다...")

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
        control_motors(angle=current_angle, speed=0, direction=None)
        print("A/D로 조향합니다. W로 전진, S로 후진합니다. 녹화를 시작하려면 'R'을 누르세요. 종료하려면 'Q'를 누르세요.")

        # 웹캠 촬영 및 저장을 별도 스레드에서 대기
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
    recording_event.set()
    root.quit()

if __name__ == "__main__":
    main()
