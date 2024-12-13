import Jetson.GPIO as GPIO
import time

# GPIO 핀 번호
servo_pin = 33  # PWM 출력이 가능한 핀

# GPIO 초기화
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

# PWM 시작 (50Hz는 서보모터에 일반적으로 사용됨)
pwm = GPIO.PWM(servo_pin, 50)

# PWM 시작
pwm.start(0)  # 0% 듀티사이클로 시작

# 서보모터 각도 설정 함수
def set_servo_angle(angle):
    try:
        # 각도를 듀티사이클로 변환
        duty_cycle = 2 + (angle / 18)
        pwm.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)
        pwm.ChangeDutyCycle(0)  # 신호를 끊어 서보모터의 떨림 방지
    except Exception as e:
        print(f"Error setting servo angle: {e}")

# 테스트 코드
try:
    for angle in range(0, 181, 30):
        print(f"Setting angle to {angle}")
        set_servo_angle(angle)
    for angle in range(180, -1, -30):
        print(f"Setting angle to {angle}")
        set_servo_angle(angle)
finally:
    pwm.stop()
    GPIO.cleanup()
