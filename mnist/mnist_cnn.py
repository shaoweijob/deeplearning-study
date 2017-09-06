#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import tensorflow as tf

'''
权重、偏置初始化
    为了这个模型，我们需要创建大量的权重和偏置项。
        1. 这个模型在初始化时，应该加入少量的噪音来打破对称性以及避免0梯度。
        2. 由于我们使用relu，因此比较好的做法是用一个较小的正数来初始化偏置项，以及避免神经元节点输出恒为0（神经元死掉）问题

'''

# w 初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# b 初始化
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

'''
卷积、池化
    TF在卷积和池化上有很强的灵活性。我们如何处理边界。步长应该设多大？在这个实例里，我们会一直使用vanilla版本。
    我们的卷积使用1步长，0边距的模板，保证输出和输入是同一个大小。
    我们的池化用简单传统的 2x2 大小模板做 max pooling。
'''
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x = tf.placeholder("float", [None, 784])

y_ = tf.placeholder("float", [None, 10])

# w = tf.Variable(tf.zeros([784, 10]))
#
# b = tf.Variable(tf.zeros([10]))

'''
第一层卷积：
    它是有一个卷积接一个max pooling完成。卷积在每个 5x5 的patch中算出32个特征。
    卷积的权重张量形状是[5, 5, 1, 32], 前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
    而对于每个输出通道都有一个对应的偏置值。
'''
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

'''
    为了这一层，我们把 x 变成一个4d向量， 其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
    （因为是灰度图，所以这里的通道数为1，如果是rgb彩色图，则为3）。
'''
x_image = tf.reshape(x, [-1, 28, 28, 1])

'''
    我们把 x_image 和 权值向量进行卷积，加上偏置项，然后应用relu激活函数，最后进行max pooling。
'''
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
第二层卷积：
    为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个 5x5 的patch会得到64个特征。
'''
W_conv2 = weight_variable([5, 5 , 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
密集连接层
    现在，图片尺寸减小到 7X7, 我们加入一个有1024个神经元的全连接层，用于处理整个图片。
    我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用relu激活函数
'''

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

'''
Dropout
    为了减少过拟合，我们在输出层之前加入dropout。
    我们用一个 placeholder 来代表一个神经元的输出在 dropout 中保持不变的概率。
    这样，我们可以在训练过程中启用 dropout，在测试过程中关闭dropout。
    TF的tf.nn.dropout 操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
    所以用dropout的时候，可以不用考虑scale。
'''

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

'''
输出层
    最后，我们添加一个softmax层
'''

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

'''
训练和评估
    用更加复杂的ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数 keep_prob 来控制dropout比例。
    然后，每100次迭代输出一次日志。
'''
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))





