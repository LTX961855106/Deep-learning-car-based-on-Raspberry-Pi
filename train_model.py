import numpy as np
from sklearn.model_selection import train_test_split
#keras是tensorflow（机器学习库）之上的高级包装器
#顺序容器是层的线性堆栈
from keras.models import Sequential
#使用梯度下降的流行优化策略
from keras.optimizers import Adam, SGD
#定期将我们的模型保存为检查点以便以后加载
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import Model, Input
import glob
import sys

#对于调试，允许重复（确定性）结果
np.random.seed(0)

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)


def load_data():
	"""
	加载训练数据并将其拆分为训练和验证集
	"""

	#加载训练数据
	image_array = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))  #初始化图片
	label_array = np.zeros((1, 5), 'float')                                 #初始化标签
	training_data = glob.glob('training_data_npz/*.npz')                    #目录

	# 如果没有数据，退出
	if not training_data:                                                  #如果没有数据
		print("No training data in directory, exit")                       #打印No training data in directory, exit
		sys.exit()                                                         #退出

	for single_npz in training_data:
		with np.load(single_npz) as data:
			train_temp = data['train_imgs']                         #定义图像
			train_labels_temp = data['train_labels']                #定义标签
		image_array = np.vstack((image_array, train_temp))          #按垂直方向（行顺序）堆叠数组构成一个新的数组
		label_array = np.vstack((label_array, train_labels_temp))   #按垂直方向（行顺序）堆叠数组构成一个新的数组

	X = image_array[1:, :]                                         #定义图像
	y = label_array[1:, :]                                         #定义标签
	print('Image array shape: ' + str(X.shape))                    #输出图像
	print('Label array shape: ' + str(y.shape))                    #输出标签
	print(np.mean(X))                                              #求取均值
	print(np.var(X))                                               #求方差

	# 现在我们可以将数据按比例分成训练集（80%）和测试集（20%）
	#从 sklearn.model_selection 中调用，https://www.cnblogs.com/Yanjy-OnlyOne/p/11288098.html
	X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0,stratify = y)
	return X_train, X_valid, y_train, y_valid


def build_model(keep_prob):
	"""
	图像标准化，以避免饱和度，并使梯度更好地工作。
	卷积核大小: 5x5, 卷积核个数: 24, 步长: 2x2, 激励函数: ELU
	卷积核大小: 5x5, 卷积核个数: 36, 步长: 2x2, 激励函数: ELU
	卷积核大小: 5x5, 卷积核个数: 48, 步长: 2x2, 激励函数: ELU
	卷积核大小: 3x3, 卷积核个数: 64, 步长: 1x1, 激励函数: ELU
	卷积核大小: 3x3, 卷积核个数: 64, 步长: 1x1, 激励函数: ELU
	Drop out (0.5)
	全连接层: 神经元个数: 100, 激励函数: ELU
	全连接层: 神经元个数: 50,  激励函数: ELU
	全连接层: 神经元个数: 10,  激励函数: ELU
	全连接层: 神经元个数: 1 (输出)

	# 卷积层是用来处理特征工程的
	# Dense就是常用的全连接层  全连接层是用来预测转向角
	dropout 避免过度适应
	ELU（指数线性单位）函数处理消失梯度问题。
	"""
	# IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 240, 240, 3
	# INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

	model = Sequential()                                             #序贯模型（Sequential):单输入单输出，一条路通到底，层与层之间只有相邻关系，没有跨层连接。这种模型编译速度快，操作也比较简单
	model.add(BatchNormalization(input_shape=INPUT_SHAPE))           #输入层
	model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))  #卷积层
	model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))  #卷积层
	model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))  #卷积层

	# model.add(Dropout(0.5))
	model.add(Conv2D(64, (3, 3), activation='elu'))                  #卷积层
	# model.add(Dropout(0.3))
	model.add(Conv2D(64, (3, 3), activation='elu'))                  #卷积层
	model.add(Dropout(keep_prob))                                    #池化层，避免过拟合
	model.add(Flatten())                                             #拍扁
	model.add(Dense(500, activation='elu'))                          #隐藏层节点500个，全连接层
	# model.add(Dropout(0.1))
	model.add(Dense(250, activation='elu'))                          #隐藏层节点250个，全连接层
	# model.add(Dropout(0.1))
	model.add(Dense(50, activation='elu'))                           #隐藏层节点50个，全连接层
	# model.add(Dropout(0.1))
	model.add(Dense(5))                                              #输出结果是5个类别，所以维度是5
	model.summary()                                                  #输出模型各层的参数状况

	return model


def train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size, X_train, X_valid, y_train, y_valid):
	"""
	训练模型
	"""
	# 在每个epoch之后保存模型。
	# 要监视的数量，详细程度，即记录模式（0或1），
	# 如果save_best_only为true，则不会覆盖根据监视数量的最新最佳型号。
	# 模式：从{auto，min，max}选一个。如果save_best_only=True，则覆盖当前保存文件的决定是
	# 基于监控量的最大化或最小化而制作。对于val_acc，
	# 这应该是最大值，对于val_损耗，这应该是最小值，等等。在自动模式下，方向是自动的
	# 从监视数量的名称推断。https://blog.csdn.net/breeze5428/article/details/80875323

	checkpoint = ModelCheckpoint(
		'model-{epoch:03d}-{loss:.4f}.h5',  #字符串，保存模型的路径
		monitor='val_loss',                 #需要监视的值
		verbose=0,                          #信息展示模式，0或1
		save_best_only=True,                #当设置为True时，监测值有改进时才会保存当前的模型
		mode='min')

	# EarlyStopping patience：当early stop 被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
	# mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
	early_stop = EarlyStopping(
		monitor='val_loss',     #需要监视的值
		min_delta=.0005,        #增大或减小的阈值
		patience=1000,
		verbose=1,
		mode='min')
	tensorboard = TensorBoard(
		log_dir='./logs',              #用来保存被 TensorBoard 分析的日志文件的文件名。
		histogram_freq=0,              #对于模型中各个层计算激活值和模型权重直方图的频率（训练轮数中）。 如果设置成 0 ，直方图不会被计算。
		batch_size=20,                 #用以直方图计算的传入神经元网络输入批的大小。
		write_graph=True,              #是否在 TensorBoard 中可视化图像。
		write_grads=True,              #是否在 TensorBoard 中可视化梯度值直方图。
		write_images=True,             #是否在 TensorBoard 中将模型权重以图片可视化。
		embeddings_freq=0,             #被选中的嵌入层会被保存的频率（在训练轮中）。
		embeddings_layer_names=None,   #一个列表，会被监测层的名字。 如果是 None 或空列表，那么所有的嵌入层都会被监测。
		embeddings_metadata=None)      #一个字典，对应层的名字到保存有这个嵌入层元数据文件的名字。

	# 计算预期转向角和实际转向角之间的差异
	# 消除差异
	# 把所有这些差异加起来，得到尽可能多的数据点
	# 除以它们的数目
	# 这个值是我们的均方误差！这就是我们想要最小化通过
	# 梯度下降
	# opt= SGD(lr=0.0001)
	model.compile(
		loss='mean_squared_error',            #计算损失
		optimizer=Adam(lr=learning_rate),     #优化器
		metrics=['accuracy'])                 #列表，包含评估模型在训练和测试时的性能的指标
	# 适用于Python编译器逐批生成的数据的模型。
	# 为了提高效率，编译器与模型并联运行。
	# 例如，这允许您做基于CPU的图像实时数据增强
	# 在GPU上训练你的模型。
	# 因此，我们将数据重新组合成适当的批，并模拟地训练我们的模型
    #fit_generator节省内存
	model.fit_generator(
		batch_generator(X_train, y_train, batch_size),                  #一个生成器，以在使用多进程时避免数据的重复。
		steps_per_epoch=samples_per_epoch/batch_size,                   # 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。
		epochs=nb_epoch,                                                #训练模型的迭代总轮数。
		max_queue_size=1,                                               #生成器队列的最大尺寸。
		validation_data=batch_generator(X_valid, y_valid, batch_size),  #在每个 epoch 结束时评估损失和任何模型指标。该模型不会对此数据进行训练。
		validation_steps=len(X_valid)/batch_size,                       #仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。
		callbacks=[tensorboard, checkpoint, early_stop],                #在训练时调用的一系列回调函数。
		verbose=2)                                                      #0, 1 或 2。日志显示模式。 0 = 安静模式, 1 = 进度条, 2 = 每轮一行。
#也可用model.fit
#    model.fit(X_train,y_train,samples_per_epoch,nb_epoch,max_q_size=1,X_valid,y_valid,\
#              nb_val_samples=len(X_valid),callbacks=[checkpoint],verbose=1)


def batch_generator(X, y, batch_size):
	"""
随机生成训练图像给出图像路径和相关的转向角
	"""
	images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])      #返回一个随机元素的矩阵，大小按照参数定义。
	steers = np.empty([batch_size,5])                                               #返回一个随机元素的矩阵，大小按照参数定义。
	while True:
		i = 0                                                #初始化计数
		for index in np.random.permutation(X.shape[0]):      #循环随机排列序列
			images[i] = X[index]                             #装入图片
			steers[i] = y[index]                             #装入标签
			i += 1                                           #计数
			if i == batch_size:                              #当装满256个图片时
				break                                        #结束循环
		yield (images, steers)                               #返回结果


def main():
	print('-' * 30)
	print('Parameters')
	print('-' * 30)

	data = load_data()                  #加载数据

	# 以下参数请自己调整测试
	keep_prob = 0.5                     #每个元素被保留的概率
	learning_rate = 0.0001              #学习率，其决定着目标函数能否收敛到局部最小值以及何时收敛到最小值。合适的学习率能够使目标函数在合适的时间内收敛到局部最小值。
	nb_epoch = 1000                     #最大迭代次数
	samples_per_epoch = len(data[0])    #每一次迭代的样本个数
	batch_size = 256                    #该demo每次提取256个样本

	print('keep_prob = {}'.format(keep_prob))
	print('learning_rate = {}'.format(learning_rate))
	print('nb_epoch = {}'.format(nb_epoch))
	print('samples_per_epoch = {}'.format(samples_per_epoch))
	print('batch_size = {}'.format(batch_size))
	print('-' * 30)

	# 创建模型
	model = build_model(keep_prob)
	# 训练模型并保存
	train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size, *data)


if __name__ == '__main__':
	main()
