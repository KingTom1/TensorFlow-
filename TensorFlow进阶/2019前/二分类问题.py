import tensorflow as tf

# Numpy 是一个科学计算工具包，这里通过Numpy工具包生成模拟数据集
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_= tf.placeholder(tf.float32,shape=(None,1),name='y-input')

# 定义神经网络前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

# 定义损失函数和反向传播的算法
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)  # 随机生成一个128行2列的二维数组，由于设定随机种子为1，所以每次运行随机生成的数组相同

Y = [[int(x1+x2 < 1)] for (x1, x2) in X]
# 创建一个会话来运行TensorFlow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size个样本进行训练
        start = (i * batch_size)%dataset_size
        end = min(start+batch_size,dataset_size)
        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        # 可以间隔一些时间，查看损失值，交叉熵越小，说明预测结果与真实值差距越小
        # if i%1000 == 0:
        #     # 每隔一段时间计算在所有数据上的交叉熵输出
        #     total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
        #
        #     print("After %d training step(s),cross entropy on all data is %g"%(i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))




