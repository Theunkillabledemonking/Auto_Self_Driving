import Jetson.GPIO as GPIO
import time
from pynput import keyboard

# GPIO 초기 설정
GPIO.setwarnings(False)
servo_pin = 33
dc_motor_pwm_pin = 32
dc_motor_dir_pin1 = 29
dc_motor_dir_pin2 = 31

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

# PWM 설정
servo = GPIO.PWM(servo_pin, 50)  # 서보 모터용 PWM 주파수 설정
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DC 모터용 1kHz PWM
servo.start(0)
dc_motor_pwm.start(0)

fixed_speed = 75
current_speed = 0
current_angle = 90

# 서보 각도 설정 함수 (듀티 사이클 범위를 제한적으로 설정)
def set_servo_angle(angle):
    if 0 <= angle <= 180:  # 각도 범위 체크
        try:
            duty_cycle = max(2.5, min(12.5, 2.5 + (angle / 18)))
            servo.ChangeDutyCycle(duty_cycle)
            time.sleep(0.05)  # 진동 방지를 위한 작은 딜레이
            servo.ChangeDutyCycle(0)  # 신호를 끄기 위해 0으로 설정
        except OSError as e:
            print(f"OSError 발생 (서보 각도 변경 실패): {e}")
    else:
        print(f"Invalid angle: {angle}")

# DC 모터 속도 및 방향 설정 함수
def set_dc_motor(speed, direction="forward"):
    GPIO.output(dc_motor_dir_pin1, GPIO.HIGH if direction == "forward" else GPIO.LOW)
    GPIO.output(dc_motor_dir_pin2, GPIO.LOW if direction == "forward" else GPIO.HIGH)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 키보드 입력 핸들러
def on_press(key):
    global current_speed, current_angle
    try:
        if key.char == 'w':
            current_speed = min(current_speed + 5, fixed_speed)
            set_dc_motor(current_speed, "forward")
            print(f"속도 증가: {current_speed}%")
        elif key.char == 's':
            current_speed = min(current_speed + 5, fixed_speed)
            set_dc_motor(current_speed, "backward")
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
            print("종료 키 입력")
            return False
    except AttributeError:
        pass

def on_release(key):
    global current_speed
    if current_speed > 0:
        current_speed = max(current_speed - 5, 0)
        set_dc_motor(current_speed, "forward")
    elif current_speed < 0:
        current_speed = min(current_speed + 5, 0)
        set_dc_motor(abs(current_speed), "backward")
    print(f"감속 중: {current_speed}%")

try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
finally:
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()

