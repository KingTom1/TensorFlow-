'''梯度下降算法主要用于优化单个参数的取值,
    反向传播算法给出了一个高效的方式在所有参数上使用梯度下降算法,从而使神经网络模型在训练数据上的损失函数尽可能小'''
# 梯度下降算法 得不到 全局最小值
# 随机梯度下降算法 优化加速训练过程
'''TensorFlow解决梯度下降问题的规范书写'''
import tensorflow as tf
batch_size = 0
# 每次读取一小部分数据作为当前的训练数据来执行反向传播算法
x = tf.placeholder(tf.float32,shape=(batch_size,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(batch_size,1),name='y-input')
# 定义神经网络结构和优化算法
loss = ...
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

# 训练神经网络
with tf.Session() as sess:
    # 参数初始化
    ...

# 迭代更新参数
STEPS = 5000
for i in range(STEPS):
    # 准备batch_size个训练数据,一般将所有训练数据随机打乱之后再选取可以得到更好的优化效果
    current_X, current_Y = ...
    sess.run(train_step,feed_dict={x: current_X, y_: current_Y})
