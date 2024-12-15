import Jetson.GPIO as GPIO
import time

# Pin ¼³Á¤
SERVO_PIN = 33  # ¼­º¸¸ðÅÍ°¡ ¿¬°áµÈ ÇÉ ¹øÈ£ (PWM Áö¿ø ÇÉ)

# GPIO ÃÊ±âÈ­
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# PWM ¼³Á¤
servo = GPIO.PWM(SERVO_PIN, 50)  # ÁÖÆÄ¼ö 50Hz
servo.start(0)  # ÃÊ±â duty cycle

def set_servo_angle(angle):
    """
    ¼­º¸¸ðÅÍ °¢µµ¸¦ ¼³Á¤ÇÏ´Â ÇÔ¼ö.
    :param angle: 0~180 »çÀÌÀÇ °¢µµ
    """
    try:
        duty_cycle = 2 + (angle / 18)  # °¢µµ¿¡ µû¶ó duty cycle °è»ê
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # ¸ðÅÍ°¡ ¿òÁ÷ÀÏ ½Ã°£À» ±â´Ù¸²
        servo.ChangeDutyCycle(0)  # ½ÅÈ£ Á¦°Å·Î ¸ðÅÍ ¾ÈÁ¤È­
    except Exception as e:
        print(f"Error setting servo angle: {e}")

try:
    print("Testing servo motor...")
    # ¼­º¸¸ðÅÍ¸¦ 0µµ¿¡¼­ 180µµ±îÁö È¸Àü
    for angle in range(0, 181, 30):
        set_servo_angle(angle)

    # ¼­º¸¸ðÅÍ¸¦ 180µµ¿¡¼­ 0µµ·Î µÇµ¹¸²
    for angle in range(180, -1, -30):
        set_servo_angle(angle)

finally:
    print("Cleaning up GPIO...")
    servo.stop()
    GPIO.cleanup()

