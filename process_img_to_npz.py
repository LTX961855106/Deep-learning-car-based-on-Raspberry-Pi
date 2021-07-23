import os
import numpy as np
import matplotlib.image as mpimg
# from PIL import Image
from time import time
import math

CHUNK_SIZE = 256

def process_img(img_path, key):
	print(img_path, key)                                #输出图片路径和key

	# 使用PIL将图像文件转换为numpy数组
	# image = Image.open(img_path)
	# image_array = np.array(image)
	# image_array = np.expand_dims(image_array,axis = 0)
	
	# 使用matplotlib将图像文件转换为numpy数组
	image_array = mpimg.imread(img_path)                #读取和代码处于同一目录下的图片
	image_array = np.expand_dims(image_array,axis = 0)  #表示在0位置添加数据
	print(image_array.shape)                            #输出

	if key == 2:
		label_array = [ 0.,  0.,  1.,  0.,  0.]         #定义各个key所对应的浮点数数组
	elif key ==3:
		label_array = [ 0.,  0.,  0.,  1.,  0.]
	elif key == 0:
		label_array = [ 1.,  0.,  0.,  0.,  0.]
	elif key == 1:
		label_array = [ 0.,  1.,  0.,  0.,  0.]
	elif key == 4:
		label_array = [ 0.,  0.,  0.,  0.,  1.]

	return (image_array, label_array)

if __name__ == '__main__':
	path = "training_data"    #训练数据路径
	files= os.listdir(path)   #返回指定的文件夹包含的文件
	turns = int(math.ceil(len(files) / CHUNK_SIZE))  #统计图片数量确立压缩循环次数
	print("number of files: {}".format(len(files)))  #显示一共有多少张图片
	print("turns: {}".format(turns))                 #显示一共要多少次循环

	for turn in range(0, turns):
		train_labels = np.zeros((1,5),'float')      #初始化图像标签
		train_imgs = np.zeros([1,120,160,3])        #初始化图像数组

		CHUNK_files = files[turn*CHUNK_SIZE: (turn+1)*CHUNK_SIZE]                   #读取文件
		print("number of CHUNK files: {}".format(len(CHUNK_files)))                 #打印一共有多少个文件
		for file in CHUNK_files:
			if not os.path.isdir(file) and file[len(file)-3:len(file)] == 'jpg':     #当前是否为文件并且是否为jpg格式
				try:
					key = int(file[0])
					image_array, label_array = process_img(path+"/"+file, key)       #读取图像和标签
					train_imgs = np.vstack((train_imgs, image_array))      #按垂直方向（行顺序）堆叠数组构成一个新的数组
					train_labels = np.vstack((train_labels, label_array))  #按垂直方向（行顺序）堆叠数组构成一个新的数组
				except Exception as e:                                     #可以捕获除与程序退出sys.exit()相关之外的所有异常
					print('prcess error: {}'.format(e))

		# 去掉第0位的全零图像数组，全零图像数组是 train_imgs = np.zeros([1,120,160,3]) 初始化生成的
		train_imgs = train_imgs[1:, :]
		train_labels = train_labels[1:, :]
		file_name = str(int(time()))           #设置npz文件命名
		directory = "training_data_npz"        #设置npz文件保存路径

		if not os.path.exists(directory):      #判断文件或文件夹是否存在
			os.makedirs(directory)             #递归创建目录
		try:    
			np.savez(directory + '/' + file_name + '.npz', train_imgs=train_imgs, train_labels=train_labels)       #保存
		except IOError as e:                   #错误退出
			print(e)                           #输出错误
