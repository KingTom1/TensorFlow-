'''学习率既不能过小也不能过大  过小导致效率降低迭代次数增加    过大导致无法得到最优解即总是在左右两边摇摆'''
# TensorFlow提供指数衰减法(exponential_decay)  解决学习率设置的问题
'decayed_learning_rate = learning_rate * decayed_rate ^ (global_step/decay_step)'

# 方法 exponential_decay(staircase=false)  默认参数 表示连续型下降衰减学习率    true 表示阶梯状衰减学习率
import tensorflow as tf
global_step = tf.Variable(0)
# 通过exponential_decay函数生成学习率  例:每100次迭代学习率乘以0.96 基础学习率为 0.1
learning_rate = tf.train.exponential_decay(0.1,global_step,100,0.96,staircase=True)
# 使用指数衰减的学习率,在minimize函数中传入global_step 将自动更新
# global_step参数,从而使得学习率也得到相应更新
learning_step = tf.train.ProximalGradientDescentOptimizer(learning_rate).minimize(...,global_step=global_step)