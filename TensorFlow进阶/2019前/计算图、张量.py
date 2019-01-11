import tensorflow as tf

a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
# print(a.graph is tf.get_default_graph())  # 输出为true

# tf.Graph函数生成新的计算图，不同计算图上的张量和运算不会共享
# g1 = tf.Graph()
# with g1.as_default():
#     # 在计算图gl中定义变量“V”，并设置初始值为0
#     v = tf.get_variable("v",shape=[1],initializer=tf.zeros_initializer)
#
# g2 = tf.Graph()
# with g2.as_default():
#     # 在计算图g2中定义变量“V”，并设置初始值为1
#     v = tf.get_variable("v",shape=[1],initializer=tf.ones_initializer)
#
# #在计算图g1中读取变量“v”的取值
# with tf.Session(graph=g1) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable('v')))
#
# #在计算图g2中读取变量“v”的取值
# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("",reuse=True):
#         print(sess.run(tf.get_variable('v')))

# tf.Graph.device函数指定运行计算的设备
# g = tf.Graph()
# with g.device('/gpu:0'):
#     result = a + b
#     print(result)

# 会话模式
# sess = tf.Session()
# with sess.as_default():
#     print(result.eval())

# 以下代码也能完成相同功能
# sess = tf.Session()
# print(sess.run(result))   # 同下结果相同
# print(result.eval(session=sess))  # 同上结果相同

# sess = tf.InteractiveSession()
# # print(result.eval())
# # sess.close()

# # 第一个参数allow_soft_placement 自动切换GPU到CPU ；  log_device_placement日志记录每个节点被安排到了哪个设备
# config = tf.ConfigProto(allow_soft_placement = True,log_device_placement = True)
# sess1 = tf.InteractiveSession(config = config)
# sess2 = tf.Session(config= config)

