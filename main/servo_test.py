import Jetson.GPIO as GPIO
import time

servo_pin = 33
GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin, GPIO.OUT)

servo = GPIO.PWM(servo_pin, 50)
servo.start(0)

try:
    while True:
        for angle in range(0, 181, 10):
            duty_cycle = 2 + (angle / 18)
            servo.ChangeDutyCycle(duty_cycle)
            time.sleep(0.5)
        for angle in range(180, -1, -10):
            duty_cycle = 2 + (angle / 18)
            servo.ChangeDutyCycle(duty_cycle)
            time.sleep(0.5)

except KeyboardInterrupt:
    pass
finally:
    servo.stop()
    GPIO.cleanup()