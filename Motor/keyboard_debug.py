import Jetson.GPIO as GPIO
import time
from pynput import keyboard

# GPIO 설정
servo_pin = 33
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

# 서보 모터 PWM 설정
servo = GPIO.PWM(servo_pin, 50)
servo.start(0)

# 기본 설정 값
current_angle = 90

# 서보 각도 설정 함수
def set_servo_angle(angle):
    try:
        duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 18)))
        servo.ChangeDutyCycle(duty_cycle)
    except OSError as e:
        print(f"OSError 발생 (서보 각도 변경 실패): {e}")

# 키보드 입력 핸들러
def on_press(key):
    global current_angle
    try:
        if key.char == 'a':
            current_angle = max(0, current_angle - 5)
            set_servo_angle(current_angle)
            print(f"좌회전 각도: {current_angle}도")
        elif key.char == 'd':
            current_angle = min(180, current_angle + 5)
            set_servo_angle(current_angle)
            print(f"우회전 각도: {current_angle}도")
        elif key.char == 'q':
            print("종료 키 입력")
            return False
    except AttributeError:
        pass

try:
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
finally:
    servo.stop()
    GPIO.cleanup()

