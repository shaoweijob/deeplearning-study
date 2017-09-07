# coding=utf-8
import tensorflow as tf

'''
    计算机视觉中，卷积的价值体现在对输入（本例中为图像）的降维的能力上。
    一幅2D的图像的维度包括其宽度、高度、通道数。如果图像具有较高的维数，
    则意味着神经网络扫描所有图像以判断各像素的重要性所需的时间呈指数级增长。
    利用卷积运算对图像降维是通过修改卷积核的strides（跨度）参数实现的。
'''

input_batch = tf.constant([
    [   # input 1:
        [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
        [[0.1], [1.1], [2.1], [3.1], [4.1], [5.1]],
        [[0.2], [1.2], [2.2], [3.2], [4.2], [5.2]],
        [[0.3], [1.3], [2.3], [3.3], [4.3], [5.3]],
        [[0.4], [1.4], [2.4], [3.4], [4.4], [5.4]],
        [[0.5], [1.5], [2.5], [3.5], [4.5], [5.5]]
    ]
])
print input_batch #  Tensor("Const:0", shape=(1, 6, 6, 1), dtype=float32)
kernel = tf.constant([
    [
        [[0.0]],[[0.5]],[[0.0]]
    ],
    [
        [[0.0]],[[1.0]],[[0.0]]
    ],
    [
        [[0.0]],[[0.5]],[[0.0]]
    ]
])

print kernel #  Tensor("Const_1:0", shape=(3, 3, 1, 1), dtype=float32)

'''
    设置跨度是一种调整输入张量维数的方法。
    降维可减少所需的运算量，并可避免创建一些完全重叠的感受域。
    strides参数的格式域输入向量相同，即(image_batch_size, image_height_stride, iamge_width_stride, image-channels_stride)
    第一个和最后一个跨度很少修改，因为它们会在tf.nn.conv2d运算中跳过一些数据。
    如果希望降低输入的维数，可修改image_height_stride和image_width_stride参数。
    
    在对输入使用跨度参数时，所面临的一个挑战是如何应对那些不是恰好在输入的边界到达尽头的跨度值。
    非均匀的跨度通常在图像尺寸和卷积核尺寸与跨度参数不匹配时出现。
    如果图像尺寸，卷积核尺寸和strides参数都无法改变，则可采取对图像填充边界的方法来处理那些非均匀区域。
'''
conv2d = tf.nn.conv2d(input_batch, kernel, strides=[1, 3, 3, 1], padding="SAME")

with tf.Session() as sess:
    print sess.run(conv2d)

'''
    卷积核与图像重叠时，它应当坐落在图像的边界内。
    有时，两者的尺寸可能不匹配，一种较好的补救策略是对图像缺失的区域进行填充，即边界填充。
    SAME：   卷积输出与输入尺寸相同。这里在计算如何跨越图像时，并不考虑滤波器的尺寸。
            选用该设置时，缺失的像素将用0填充，卷积核扫过的像素数超过图像的实际像素数
            
    VALID：  在计算卷积核如何在图像上跨越时，需要考虑滤波器的尺寸。这会使卷积核尽量不越过图像的边界。
            在某些情形下，可能边界也会被填充。
            
    在计算卷积时，最好能够考虑图像的尺寸，如果边界填充是必要的，则TF会有一些内置选项。
    在大多数比较简单的情形下，SAME都是一个不错的选择。
    当指定跨度参数后，如果输入和卷积核能够很好地工作，则推荐使用VILID。
'''

