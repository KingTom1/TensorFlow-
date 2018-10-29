import tensorflow as tf

# 类型不匹配报错  dtype=tf.float64
# w1 = tf.Variable(tf.random_normal([2,3],stddev=1),name="w1")
# w2 = tf.Variable(tf.random_normal([2,3],stddev=1,dtype=tf.float64),name="w2")
# # w1.assign(w2)

# 维度不匹配报错，2行3列  和  2行2列
# w1 = tf.Variable(tf.random_normal([2,3],stddev=1),name="w1")
# w2 = tf.Variable(tf.random_normal([2,2],stddev=1),name="w2")
# tf.assign(w1,w2)