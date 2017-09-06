import tensorflow as tf

# a = tf.Variable(tf.random_normal([2,2]))
b = tf.Variable(tf.truncated_normal([2,2],stddev=0.1,seed=1))
c = tf.Variable(tf.truncated_normal([2,2],stddev=0.1,seed=1))
d = tf.Variable(tf.constant(1, shape = [2,2]))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))