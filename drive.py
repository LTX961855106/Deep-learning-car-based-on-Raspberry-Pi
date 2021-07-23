import os
import io
import glob
import time
import threading
import picamera
import picamera.array
from PIL import Image
import numpy as np

import car_control
from keras.models import load_model
import tensorflow as tf
	
# def get_max_prob_num(predictions_array):
# 	"""得到整数的可能性"""
	
# 	prediction_edit = np.zeros([1,5])                             #初始化预测数组
# 	for i in range(0,5):                                          #循环
# 		if predictions_array[0][i] == predictions_array.max():    #获得预测的最大可能性
# 			prediction_edit[0][i] = 1                             #将最大可能性改为1
# 			return i
# 	return 2

def control_car(action_num):
	"""此处用到car_control"""

	if action_num == 0:                   #如果动作标识为2
		print("Left")                     #打印左转
		car_control.car_turn_left()       #控制小车左转
		time.sleep(0.25)                  #延时
	elif action_num== 1:                  #如果动作标识为1
		print("Right")                    #打印右转
		car_control.car_turn_right()      #控制小车右转
		time.sleep(0.25)                  #延时
	elif action_num == 2:                 #如果动作标识为2
		car_control.car_turn_straight()   #控制小车回直
		car_control.car_move_forward()    #控制小车前进
		print('Forward')                  #打印前进
	elif action_num == 3:                 #如果动作标识为3
		car_control.car_move_backward()   #控制小车后退
		print('Backward')                 #打印后退
	else:                                 #动作标识不为0、1、2、3
		car_control.car_stop()            #控制小车停止
		print('Stop')                     #打印停止

#https://blog.csdn.net/xqf1528399071/article/details/52850157
class ImageProcessor(threading.Thread):
	def __init__(self, owner):
		super(ImageProcessor, self).__init__()  #把父类的 __init__构造方法拿过来用, 并且可以对父类的__init__方法进行补充(比如添加成员属性/方法) ,也就相当于把父类的__init__方法继承过来了
		self.stream = io.BytesIO()              #在内存中读写bytes
		self.event = threading.Event()          #创建一个事件管理标志
		self.terminated = False                 #停止
		self.owner = owner
		self.start()                            #开启

	def run(self):
		global latest_time, model, graph                                     #将变量定义为全局变量
		# 此方法在单独的线程中运行
		while not self.terminated:                                           #当self.terminated = True

			if self.event.wait(1):                                           #等待图像写入流
				try:
					self.stream.seek(0)                                      #执行图片的处理过程
					# 加载图像并对其进行处理
					image = Image.open(self.stream)                          #打开图片
					image_np = np.array(image)                               #将图片转为np格式
					camera_data_array = np.expand_dims(image_np,axis = 0)    #表示在0位置添加数据
					current_time = time.time()                               #记录时间
					if current_time>latest_time:                             #如果当前时间大于过去时间
						if current_time-latest_time>1:                       #如果当前时间比过去时间大于1
							print("*" * 30)
							print(current_time-latest_time)                  #打印时间
							print("*" * 30)
						latest_time = current_time                           #更新过去时间
						with graph.as_default():                             #返回图片
							predictions_array = model.predict(camera_data_array, batch_size=20, verbose=1)  #带入模型获得预测结果
						print(predictions_array)                             #输出模型结果
						# action_num = get_max_prob_num(predictions_array)
						action_num = predictions_array.argmax()              #返回最大的那个数值所在的下标
						control_car(action_num)                              #输出动作标识
					# 如果要将预测为名称的图像保存，请取消注释此行，但这可能会导致延迟
						# image.save('%s_image%s.jpg' % (action_num,time.time()))
				finally:
					# 处理完成，释放所有处理序列
					self.stream.seek(0)              #执行图片的处理过程
					self.stream.truncate()           #清除流
					self.event.clear()               #清除事件
					# 将处理完的图片加载到序列中
					with self.owner.lock:
						self.owner.pool.append(self)

class ProcessOutput(object):
	def __init__(self):
		self.done = False                #程序没有结束
		# 构造一个由4个图像处理器和一个lock组成的pool
		# 控制线程之间的访问
		self.lock = threading.Lock()
		self.pool = [ImageProcessor(self) for i in range(4)]
		self.processor = None           #初始化处理器

	def write(self, buf):
		if buf.startswith(b'\xff\xd8'):
			# 新帧；设置当前处理器并获取

			if self.processor:
				self.processor.event.set()             #开启事件
			with self.lock:
				if self.pool:
					self.processor = self.pool.pop()   #从尾部弹出一个pool
				else:
					# 没有可用的处理器，将跳过
					self.processor = None
		if self.processor:
			self.processor.stream.write(buf)           #写入

	def flush(self):
		# 当录制结束时，关闭

		# 回到pool
		if self.processor:
			with self.lock:
				self.pool.append(self.processor)   #添加当前处理器
				self.processor = None
		# 清空pool中的所有线程
		while True:
			with self.lock:
				try:
					proc = self.pool.pop()   #从尾部弹出一个pool
				except IndexError:
					pass                     #pool空了
			proc.terminated = True           #结束
			proc.join()                      #将序列中的元素以指定的字符连接生成一个新的字符串


def main():
	"""获取数据，然后预测数据，编辑数据，然后控制汽车"""
	global model, graph                                            #将变量定义为全局变量
	
	model_loaded = glob.glob('model/*.h5')                         #载入模型
	for single_mod in model_loaded:
		model = load_model(single_mod)                             #获取模型
	graph = tf.get_default_graph()                                 #获取当前默认的计算图
	
	try:
		with picamera.PiCamera(resolution=(160,120)) as camera:    #设置图像大小
			# 取消对此行的注释，相机图像将颠倒
			# camera.vflip = True
			time.sleep(2)                                          #延时
			output = ProcessOutput()                               #获取进程中的图像
			camera.start_recording(output, format='mjpeg')         #保存格式
			while not output.done:                                 #循环
				camera.wait_recording(1)                           #若没停止等
			camera.stop_recording()                                #若停止关闭摄像头
	finally:
		car_control.cleanGPIO()                                    #清除端口

if __name__ == '__main__':
	global latest_time                                             #将变量定义为全局变量
	latest_time = time.time()                                      #记录结束时间
	main()                                                         #主函数
