import Jetson.GPIO as GPIO
import time
import keyboard

# DC 모터 핀 설정
dc_motor_pwm_pin = 32  # DC 속도 제어용 PWM 핀
dc_motor_dir_pin1 = 29  # DC 모터 방향 제어 핀 1
dc_motor_dir_pin2 = 31  # DC 모터 방향 제어 핀 2

# 서보 모터 핀 설정
servo_pin = 33  # 서보 모터 제어용 PWM 핀
servo_gnd_pin = 34  # 서보 모터 GND 핀

# GPIO 설정
GPIO.setmode(GPIO.BOARD)
GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)
GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(servo_gnd_pin, GPIO.OUT)
GPIO.output(servo_gnd_pin, GPIO.LOW)  # 서보 모터 GND 설정

# PWM 설정
dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # 1kHz 주파수의 PWM 생성
servo = GPIO.PWM(servo_pin, 50)  # 서보 모터용 50Hz 주파수의 PWM 생성
dc_motor_pwm.start(0)  # DC 모터의 초기 속도 0%
servo.start(7.5)  # 서보 모터의 초기 위치 90도(중앙)

# 서보 모터 초기 각도
current_servo_angle = 90
servo_speed = 10  # 서보 모터가 움직일 각도 단위 (좌우로 10도씩)

# DC 모터 제어 함수
def set_dc_motor(speed, direction):
    # 방향 설정
    if direction == "forward":
        GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
        GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
        GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
    
    # 속도 설정 (0에서 100% 사이의 듀티 사이클)
    dc_motor_pwm.ChangeDutyCycle(speed)

# 서보 모터 각도 제어 함수
def set_servo_angle(angle):
    global current_servo_angle
    if 0 <= angle <= 180:  # 각도가 서보 모터 허용 범위 내에 있는 경우
        duty_cycle = 2 + (angle / 18)
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.3)  # 서보 모터가 목표 위치에 도달할 시간을 줌
        servo.ChangeDutyCycle(0)  # 신호를 끊어 안정적으로 위치 유지
        current_servo_angle = angle

try:
    print("방향키로 DC 모터와 서보 모터를 제어할 수 있습니다.")
    print("위쪽/아래쪽 방향키: DC 모터 전진/후진")
    print("왼쪽/오른쪽 방향키: 서보 모터 좌우 회전")
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
        
        # 왼쪽 방향키를 누르면 서보 모터를 좌측으로 회전
        if keyboard.is_pressed('left'):
            set_servo_angle(current_servo_angle - servo_speed)
        # 오른쪽 방향키를 누르면 서보 모터를 우측으로 회전
        elif keyboard.is_pressed('right'):
            set_servo_angle(current_servo_angle + servo_speed)
            
        # 'q' 키를 누르면 종료
        if keyboard.is_pressed('q'):
            break
        
        time.sleep(0.1)  # 너무 빠른 루프 실행을 방지하기 위한 딜레이

finally:
    # 모든 PWM을 정지하고 GPIO 정리
    dc_motor_pwm.stop()
    servo.stop()
    GPIO.cleanup()
