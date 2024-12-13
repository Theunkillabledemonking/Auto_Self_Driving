import Jetson.GPIO as GPIO
import time
import keyboard

# L298N 모터 드라이버에 연결된 핀 설정
enA = 32  # 속도 제어용 PWM 핀 (Enable A 핀)
in1 = 29  # 방향 제어 핀 1
in2 = 31  # 방향 제어 핀 2

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setup(enA, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)

# PWM 설정
motor_pwm = GPIO.PWM(enA, 1000)  # 1kHz 주파수의 PWM 생성
motor_pwm.start(0)  # 모터의 초기 속도 0%

# DC 모터 제어 함수
def set_dc_motor(speed, direction):
    # 방향 설정
    if direction == "forward":
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
    
    # 속도 설정 (0에서 100% 사이의 듀티 사이클)
    motor_pwm.ChangeDutyCycle(speed)

try:
    print("방향키로 DC 모터를 제어할 수 있습니다.")
    print("위쪽/아래쪽 방향키: DC 모터 전진/후진")
    print("종료하려면 'q'를 누르세요.")

    while True:
        # 위쪽 방향키를 누르면 DC 모터 전진
        if keyboard.is_pressed('up'):
            set_dc_motor(50, "forward")
        # 아래쪽 방향키를 누르면 DC 모터 후진
        elif keyboard.is_pressed('down'):
            set_dc_motor(50, "backward")
        # 방향키가 눌리지 않으면 DC 모터 멈춤
        else:
            set_dc_motor(0, "forward")

        # 'q' 키를 누르면 종료
        if keyboard.is_pressed('q'):
            break

finally:
    # 모든 PWM을 정지하고 GPIO 정리
    motor_pwm.stop()
    GPIO.cleanup()
