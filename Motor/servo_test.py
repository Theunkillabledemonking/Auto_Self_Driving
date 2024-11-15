import Jetson.GPIO as GPIO
import time

# GPIO 설정
servo_pin = 33
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50)  # 서보 모터용 50Hz PWM
servo.start(0)

try:
    # 듀티 사이클 범위 테스트
    print("듀티 사이클 2.5%")
    servo.ChangeDutyCycle(2.5)
    time.sleep(1)
    print("듀티 사이클 5%")
    servo.ChangeDutyCycle(5)
    time.sleep(1)
    print("듀티 사이클 7.5%")
    servo.ChangeDutyCycle(7.5)
    time.sleep(1)
    print("듀티 사이클 10%")
    servo.ChangeDutyCycle(10)
    time.sleep(1)
    print("듀티 사이클 12.5%")
    servo.ChangeDutyCycle(12.5)
    time.sleep(1)
    print("테스트 완료")

finally:
    servo.stop()
    GPIO.cleanup()

