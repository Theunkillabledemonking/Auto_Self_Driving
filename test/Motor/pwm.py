import Jetson.GPIO as GPIO
import time

SERVO_PIN = 33  # 사용 중인 GPIO 핀 번호

GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz 주파수 설정
pwm.start(7.5)  # 중립 위치 (0도)

try:
    while True:
        pwm.ChangeDutyCycle(5)   # 90도
        time.sleep(1)
        pwm.ChangeDutyCycle(10)  # 0도
        time.sleep(1)
except KeyboardInterrupt:
    pwm.stop()
    GPIO.cleanup()

