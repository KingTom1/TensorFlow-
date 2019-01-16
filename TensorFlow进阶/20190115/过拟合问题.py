'''避免过拟合   常用的方式就是正则化
    损失函数为J(θ)  那么优化时不是直接优化J(θ)  而是优化 J(θ)+λR(w) 其中λ表示模型复杂损失在总损失中的比例  R(w)刻画的是模型的复杂程度'''

import tensorflow as tf

# x = ...
# y_ = ...
# w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
# y = tf.matmul(x, w)
#
# λ = 1.0
# loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(λ)(w)

weight = tf.constant([[1.0,-2.0],[-3.0,4.0]])
with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(.5))(weight))

#contrib模块在20190116停库  需要等TensorFlow2出版再用