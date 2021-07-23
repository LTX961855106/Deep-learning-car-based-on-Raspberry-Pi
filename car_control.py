import RPi.GPIO as GPIO
import time

#定义连接的管脚
backMotorInput_ahead = 7
backMotorInput_back = 11

frontMotorInput_left = 13
frontMotorInput_right = 15

backMotorEn = 12
frontMotorEn = 16

speed = 66                                      #设置PWM占空比

GPIO.setmode(GPIO.BOARD)
#端口初始化，设置为输出模式
GPIO.setup(backMotorInput_ahead,GPIO.OUT)
GPIO.setup(backMotorInput_back,GPIO.OUT)
GPIO.setup(frontMotorInput_left,GPIO.OUT)
GPIO.setup(frontMotorInput_right,GPIO.OUT)
GPIO.setup(frontMotorEn,GPIO.OUT)
GPIO.setup(backMotorEn,GPIO.OUT)
backMotorPwm = GPIO.PWM(backMotorEn,100)
backMotorPwm.start(0)

#前进
def car_move_forward():
	GPIO.output(backMotorEn,GPIO.HIGH)           #设置为高电平
	GPIO.output(backMotorInput_ahead,GPIO.HIGH)  #设置为高电平
	GPIO.output(backMotorInput_back,GPIO.LOW)    #设置为低电平
	backMotorPwm.ChangeDutyCycle(speed)          #设置PWM调速
#后退
def car_move_backward():
	GPIO.output(backMotorEn,GPIO.HIGH)           #设置为高电平
	GPIO.output(backMotorInput_back,GPIO.HIGH)   #设置为高电平
	GPIO.output(backMotorInput_ahead,GPIO.LOW)   #设置为低电平
	backMotorPwm.ChangeDutyCycle(speed)          #设置PWM调速
#左拐
def car_turn_left():
	GPIO.output(frontMotorEn,GPIO.HIGH)          #设置为高电平
	GPIO.output(frontMotorInput_right,GPIO.LOW)  #设置为低电平
	GPIO.output(frontMotorInput_left,GPIO.HIGH)  #设置为高电平
#右拐
def car_turn_right():
	GPIO.output(frontMotorEn,GPIO.HIGH)          #设置为高电平
	GPIO.output(frontMotorInput_left,GPIO.LOW)   #设置为低电平
	GPIO.output(frontMotorInput_right,GPIO.HIGH) #设置为高电平
#停车1
def car_stop():
	GPIO.output(backMotorInput_ahead,GPIO.LOW)  #设置为低电平
	GPIO.output(backMotorInput_back,GPIO.LOW)   #设置为低电平

#停车2
def car_turn_straight():
	GPIO.output(frontMotorEn,GPIO.LOW)          #设置为低电平


#清除端口
def cleanGPIO():
	GPIO.cleanup()               #清除端口
	backMotorPwm.stop()          #停止PWM

if __name__ == '__main__':
	car_move_forward()          #前进
	time.sleep(2)               #延时
	car_move_backward()         #后退
	time.sleep(2)               #延时
	car_turn_left()             #左转
	time.sleep(2)               #延时
	car_turn_right()            #右转
	time.sleep(2)               #延时
