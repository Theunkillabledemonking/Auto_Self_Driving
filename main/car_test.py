import os
import Jetson.GPIO as GPIO
import time

# Pin configuration
SERVO_PIN = 33  # The GPIO pin connected to the servo motor (PWM signal)

# Function to run shell commands
def run_command(cmd):
    """
    Executes a system command.
    :param cmd: The command string to execute.
    """
    try:
        os.system(cmd)
    except Exception as e:
        print(f"Error running command '{cmd}': {e}")

# Register configuration commands
def configure_hardware_registers():
    """
    Configure hardware registers using devmem for specific GPIO/pin settings.
    """
    commands = [
        "busybox devmem 0x700031fc 32 0x45",  # Example hardware configuration
        "busybox devmem 0x6000d504 32 0x2",
        "busybox devmem 0x70003248 32 0x46",
        "busybox devmem 0x6000d100 32 0x00"
    ]
    for cmd in commands:
        run_command(cmd)

# GPIO initialization
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN, GPIO.OUT)

# PWM setup
servo = GPIO.PWM(SERVO_PIN, 50)  # PWM frequency: 50Hz
servo.start(0)  # Initialize with a duty cycle of 0

def set_servo_angle(angle):
    """
    Adjust the servo motor angle.
    :param angle: The angle to set (range: 0 to 180 degrees)
    """
    try:
        duty_cycle = 2 + (angle / 18)  # Calculate duty cycle for the angle
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.5)  # Allow the servo to reach the target position
        servo.ChangeDutyCycle(0)  # Stop sending signal after adjustment
    except Exception as e:
        print(f"Error setting servo angle: {e}")

try:
    print("Configuring hardware registers...")
    configure_hardware_registers()  # Execute register configuration

    print("Testing servo motor...")
    # Rotate the servo motor from 0° to 180° in steps of 30°
    for angle in range(0, 181, 30):
        set_servo_angle(angle)

    # Rotate the servo motor back from 180° to 0°
    for angle in range(180, -1, -30):
        set_servo_angle(angle)

finally:
    print("Cleaning up GPIO...")
    servo.stop()
    GPIO.cleanup()
