import Jetson.GPIO as GPIO
import time

# 핀 번호 설정
SERVO_PIN = 33  # 서보모터용 PWM 핀
DC_MOTOR_PWM_PIN = 32  # DC 모터 속도 제어용 PWM 핀
DC_MOTOR_DIR_PIN1 = 29  # DC 모터 방향 제어 핀 1
DC_MOTOR_DIR_PIN2 = 31  # DC 모터 방향 제어 핀 2

# GPIO 초기화
GPIO.setwarnings(False)  # GPIO 경고 비활성화
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN1, GPIO.OUT)
GPIO.setup(DC_MOTOR_DIR_PIN2, GPIO.OUT)

# PWM 초기화
servo = GPIO.PWM(SERVO_PIN, 50)  # 서보모터 주파수 50Hz
dc_motor_pwm = GPIO.PWM(DC_MOTOR_PWM_PIN, 1000)  # DC 모터 주파수 1kHz
servo.start(0)  # 서보 초기화
dc_motor_pwm.start(0)  # DC 모터 초기화

def set_servo_angle(angle):
    """
    서보모터 각도를 설정하는 함수.
    :param angle: 0~180 사이의 각도
    """
    try:
        duty_cycle = max(2, min(12, 2 + (angle / 18)))  # 듀티 사이클을 2%~12%로 제한
        print(f"Setting servo angle to {angle}° (Duty Cycle: {duty_cycle:.2f}%)")
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # 모터가 움직일 시간을 기다림
        servo.ChangeDutyCycle(0)  # 신호 제거로 안정화
    except Exception as e:
        print(f"Error setting servo angle: {e}")

def set_dc_motor(speed, direction):
    """
    DC 모터 속도와 방향을 설정하는 함수.
    :param speed: 0~100 (속도 %)
    :param direction: 'forward' 또는 'backward'
    """
    try:
        if direction == "forward":
            GPIO.output(DC_MOTOR_DIR_PIN1, GPIO.HIGH)
            GPIO.output(DC_MOTOR_DIR_PIN2, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(DC_MOTOR_DIR_PIN1, GPIO.LOW)
            GPIO.output(DC_MOTOR_DIR_PIN2, GPIO.HIGH)
        else:
            print(f"Invalid direction: {direction}")
            return

        print(f"Setting DC motor speed to {speed}% and direction to {direction}")
        dc_motor_pwm.ChangeDutyCycle(speed)
    except Exception as e:
        print(f"Error setting DC motor: {e}")

try:
    print("Testing servo motor...")
    # 서보모터를 0도에서 180도까지 회전
    for angle in range(0, 181, 30):
        set_servo_angle(angle)

    # 서보모터를 180도에서 0도로 되돌림
    for angle in range(180, -1, -30):
        set_servo_angle(angle)

    print("Testing DC motor...")
    # DC 모터를 50% 속도로 전진
    set_dc_motor(50, "forward")
    time.sleep(2)

    # DC 모터를 75% 속도로 후진
    set_dc_motor(75, "backward")
    time.sleep(2)

    # DC 모터 정지
    set_dc_motor(0, "forward")
    print("DC motor stopped.")

finally:
    print("Cleaning up GPIO...")
    servo.stop()
    dc_motor_pwm.stop()
    GPIO.cleanup()
