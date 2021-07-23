import io
import car_control
import os
os.environ['SDL_VIDEODRIVE'] = 'x11'
import pygame
from time import ctime,sleep,time
import threading
import numpy as np
import picamera
import picamera.array

global train_labels, train_img, is_capture_running, key                        #将变量定义为全局变量
#实时保存每张照片
class SplitFrames(object):
# 初始化格式
    def __init__(self):
        self.frame_num = 0                                                     #初始化计数
        self.output = None                                                     #初始化输出

    def write(self, buf):
        global key                                                             #将变量定义为全局变量
        if buf.startswith(b'\xff\xd8'):                                        #用于检查字符串是否是以指定子字符串开头,如果是则返回 True,否则返回 False


            if self.output:
                self.output.close()                                            #开始新的一帧；关闭旧的一帧，如果有的话
            self.frame_num += 1                                                #进行计数
            self.output = io.open('{}_image{}.jpg'.format(key,time()), 'wb')   #打开新输出
        self.output.write(buf)                                                 #写入输出
    

def pi_capture():
    global train_img, is_capture_running,train_labels,key                      #将变量定义为全局变量
    
    #初始化小车标签阵列
    print("Start capture")                                                     #打印开始摄像
    is_capture_running = True                                                  #相机开始工作

    with picamera.PiCamera(resolution=(160, 120), framerate=30) as camera:     #每帧大小和速率
        # 根据摄像头实际情况判断是否要加这句上下翻转
        # camera.vflip = True

        sleep(2)                                                               #等待相机开启
        output = SplitFrames()                                                 #读取图片
        start = time()                                                         #记录开启的时间
        camera.start_recording(output, format='mjpeg')                         #保存格式
        camera.wait_recording(120)                                             #启动开始录制,等待总共录制的时间为120s
        camera.stop_recording()                                                #停止视频的录制功能
        finish = time()                                                        #记录结束的时间
    print('Captured {} frames at {}fps'.format(
        output.frame_num,
        output.frame_num / (finish - start)))
    
    print("quit pi capture")                                                   #打印停止摄像
    is_capture_running = False                                                 #相机停止工作

def my_car_control(): 
    global is_capture_running, key #将变量定义为全局变量
    key = 4                        #初始为stop，所以定义key为4
    pygame.init()                  #初始化所有导入的pygame模块。
    pygame.display.set_mode((1,1)) #初始化窗口或屏幕以进行显示
    car_control.car_stop()         #停止小车
    sleep(0.1)                     #延时0.1秒
    print("Start control!")        #打印开始控制
 
    while is_capture_running:
        # 手动控制车辆
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key_input = pygame.key.get_pressed()                                                    #键被按下
                print(key_input[pygame.K_w], key_input[pygame.K_a], key_input[pygame.K_d])              #K_w是一个常量，库中定义好的，对应键盘的w键
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:   #w被按下，且a和d没被按下
                    print("Forward")                                                                    #打印向前
                    key = 2                                                                             #记录key=2
                    car_control.car_move_forward()                                                      #控制小车向前
                elif key_input[pygame.K_a]:                                                             #a键被按下
                    print("Left")                                                                       #打印向左
                    car_control.car_turn_left()                                                         #控制小车向左
                    sleep(0.1)                                                                          #延时0.1秒
                    key = 0                                                                             #记录key=0
                elif key_input[pygame.K_d]:                                                             #d键被按下
                    print("Right")                                                                      #打印向右
                    car_control.car_turn_right()                                                        #控制小车向右
                    sleep(0.1)                                                                          #延时0.1秒
                    key = 1                                                                             #记录key=1
                elif key_input[pygame.K_s]:                                                             #s键被按下
                    print("Backward")                                                                   #打印后退
                    car_control.car_move_backward()                                                     #控制小车向后
                    key = 3                                                                             #记录key=3
                elif key_input[pygame.K_k]:                                                             #k键被按下
                    car_control.car_stop()                                                              #控制小车停止
            elif event.type == pygame.KEYUP:                                                            #当键盘被松开
                key_input = pygame.key.get_pressed()
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:   #w被按下，且a和d没被按下
                    print("Forward")                                                                    #打印向前
                    key = 2                                                                             #记录key=2
                    car_control.car_turn_straight()                                                     #控制小车回直
                    car_control.car_move_forward()                                                      #控制小车向前
                elif key_input[pygame.K_s] and not key_input[pygame.K_a] and not key_input[pygame.K_d]: #s被按下，且a和d没被按下
                    print("Backward")                                                                   #打印后退
                    key = 3                                                                             #记录key=3
                    car_control.car_move_backward()                                                     #控制小车向后
                else:                                                                                   #当没有指令时
                    print("Stop")                                                                       #打印停止
                    car_control.car_stop()                                                              #控制小车停止
                #car_control.cleanGPIO()
    car_control.cleanGPIO()                                                                             #清除端口

if __name__ == '__main__':
    global train_labels, train_img, key                                                                 #将变量定义为全局变量

    print("capture thread")
    print ('-' * 50)
    capture_thread = threading.Thread(target=pi_capture,args=())  #创建一个线程
    capture_thread.setDaemon(True)                                #设置为守护线程,在没有用户线程可服务时会自动离开
    capture_thread.start()                                        #启动刚刚创建的线程
    
    my_car_control()                                              #开启控制

    while is_capture_running:                                     #询问相机是否还在工作
        pass                                                      #如果相机不在工作，退出循环

    print("Done!")                                                #打印结束
    car_control.car_stop()                                        #控制小车停止
    car_control.clean_GPIO()                                      #清除端口
