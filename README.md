# Deep-learning-car-based-on-Raspberry-Pi
Deep learning car based on Raspberry Pi using Tensorflow.

# How to start
## car_control.py：
引脚初始化
## collect_data.py
数据采集
将control.py和colletdata.py放在Raspberry-Pi中的一个文件夹内，运行colletdata.py进行数据采集，采集到的图片数据会保存到当前文件夹下
## process_img_to_npz.py
将图片数据压缩成npz格式
将采集的图片数据传到电脑，运行process_img_to_npz.py
## train_model.py
训练模型
## drive.py
推理与小车运行
将car_control.py和drive.py和放置权重的model文件夹放在Raspberry-Pi的同一个文件夹下，运行drive.py实现自动驾驶
