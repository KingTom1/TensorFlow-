'''Tensorflow专门提供了一个类 读取MNIST数据集和处理成需要的数据格式'''

from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集,如果本地指定路径没有下载好的数据,将自动下载MNIST数据集
mnist = input_data.read_data_sets("/path/to/MNIST_data/",one_hot=True)

# 打印训练数据 Training data size: 55000
print("Training data size: ",mnist.train.num_examples)

# 打印验收数据 Validating data size: 5000
print("Validating data size: ",mnist.validation.num_examples)
