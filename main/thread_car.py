import Jetson.GPIO as GPIO
import time
from pynput import keyboard
import cv2
import threading
import datetime

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)

# 핀 번호 설정
servo_pin = 33
dc_motor_pwm_pin = 32
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

# GPIO 핀 설정
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# pwm 설정
servo = GPIO.PWM(servo_pin, 50)
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)
servo.start(0)
dc_motor_pwm.start(0)

current_speed = 0
current_angle = 90

# 종료 신호를 위한 이벤트
stop_event = threading.Event()

def set_servo_angle(angle):
    duty_cycle = 2 + (angle / 18)
    if duty_cycle < 2 or duty_cycle > 12:
        print(f"Invalid duty cycle: {duty_cycle}")
        return
    try:
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)
        servo.ChangeDutyCycle(0)
    except Exception as e:
        print(f"Error setting servo angle: {e}")
        
def set_dc_motor(speed, direction="forward"):
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
        
def record_video(stop_event):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera")
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{timestamp}.avi"
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
        cv2.destroyAllWindows()
        print(f"Video saved as {video_filename}")
    except Exception as e:
        print(f"Error in record_video: {e}")
        
# 키보드 이벤트 핸들러
def on_press(key):
    global current_speed, current_angle
    try:
        if key.char == 'w':
            current_speed = min(current_speed + 5, 75)
            set_dc_motor(current_speed, "forward")
            print(f"속도 증가: {current_speed}%")
        elif key.char == 's':
            current_speed = max(current_speed - 5, 0)
            direction = "forward" if current_speed > 0 else "backward"
            set_dc_motor(current_speed, direction)
            print(f"속도 감소: {current_speed}%")
        elif key.char == 'a':
            current_angle = max(0, current_angle - 5)
            set_servo_angle(current_angle)
            print(f"조향 각도: {current_angle}도 (좌회전)")
        elif key.char == 'd':
            current_angle = min(180, current_angle + 5)
            set_servo_angle(current_angle)
            print(f"조향 각도: {current_angle}도 (우회전)")
        elif key.char == 'q':
            print("종료 키 입력됨")
            stop_event.set()
            return False
    except AttributeError:
        pass

try:
    set_servo_angle(current_angle)
    print("W/S로 속도 조절, A/D로 조향합니다. 종료하려면 'Q'를 누르세요")
    
    # 비디오 녹화 스레드 시작
    video_thread = threading.Thread(target=record_video, args=(stop_event,))
    video_thread.start()
    
    # 키보드 리스너 시작
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
        
    # 비디오 스레드 종료 대기
    video_thread.join()
    
except Exception as e:
    print(f"Unexpected error: {e}")
    
finally:
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
    print("GPIO 정리 및 녹화 중지 완료")