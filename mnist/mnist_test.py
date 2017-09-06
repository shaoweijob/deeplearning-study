#coding=utf=-8

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

import tensorflow as tf

# 首先载入TensorFlow库，并创建一个新的InteractiveSession,
# 使用这个命令会将这个Session注册为默认的Session，之后的运算也默认跑在这个Session中，
# 不通的session之间的数据和运算应该都是独立的

# sess = tf.Session()
# 接下来，创建一个placeholder，即输入数据的地方。
# placeholder第一个参数是数据类型，第二个参数[None, 784]代表tensor的shape
# None代表不限条数的输入，784代表每条输入是一个784维的向量

x = tf.placeholder(tf.float32, [None, 784])


# 接下来要给softmax regression模型的weights 和biases创建variable对象，第一章中提到variable是用来存储模型对象的。
# 不同于存储类型tensor一旦用掉就会消失，variable在模型训练迭代中是持久的（比如一直存放在显存中），它可以长期存在并且在每轮迭代中更新。

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# softmax 是 tf.nn下面的一个函数，而tf.nn则包含了大量神经网络的组件，tf.matmul是tensorflow中矩阵乘法函数
# tensorflow 最厉害的地方不是 定义公式，而是将forward和backward的内容都自动实现（无论CPU、GPU），只要接下来定义好loss，
# 训练时，将会自动求导并进行梯度下降，完成对softmax regression模型参数的自动学习。

y = tf.nn.softmax(tf.matmul(x, w) + b)

# 为了训练模型，我们需要定义一个loss function来描述模型对问题的分类精度。
# loss越小，代表模型的分类结果与真实值的偏差越小，也就说明模型越精确、

# 定义cross-entropy

# 先定义一个placeholder ，输入真实的label，用来计算cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 梯度下降

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 使用tensorFlow的全局参数初始化器 tf.global_variables_initializer, 并直接运行他的run方法

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# init = tf.initialize_all_variables()
# sess.run(init)
# 最后一步，我们开始迭代地执行训练操作 train_step
# 每次从训练集中抽取100条样本构成一个mini-batch，并feed给placeholder，然后调用train_step 对这些样本进行训练。
# 使用一小部分样本进行训练称为随机梯度下降，与每次使用全部样本的传统梯度下降对应

# 如果每次训练都使用全部样本，计算量太大，有时也不容易跳出局部最优。
# 因此，对于大部分机器学习问题，我们都只使用一小部分数据进行随机梯度下降，这种做法绝大多数会比全样本的收敛速度快很多
#
for i in range(1000):
    batch_xs ,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# tf.argmax 是从一个tensor中寻找最大值的序号，tf.argmax(y, 1)就是求各个预测数字中概率最大的那一个
# ，而tf.argmax(y_, 1)则是找样本的真实数字类别
# 而tf.equal方法则用来判断预测的数字类别是否就是正确的类别，最后返回计算分类是否正确的操作 correct_predition

# 我们统计全部样本预测的accuracy，这里需要先用tf.cast将之前correct_prediction输出的bool值转换为float32，再求平均

correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 我们将测试数据的特征和label输入评测流程accuracy计算模型在测试集上的准确率，再将结果打印出来。
# 使用Softmax Regression对MNIST数据进行分类识别，在测试集上平均准确率可达到92% 左右

# print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print("end")