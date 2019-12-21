from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# 载入数据
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


x = tf.placeholder(tf.float32, [None, 784])     # x为特征
# 占位符，任意张28*28的图像铺成784维向量，值为32位浮点

y = tf.placeholder(tf.float32, [None, 10])     # y为label

# Variable表示可修改的张量
W = tf.Variable(tf.zeros([784,10])) # 参数，全连接网络
b = tf.Variable(tf.zeros([10]))     # bias

# 模型的形式
prediction = tf.nn.softmax(tf.matmul(x, W)+b)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))

# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# 初始化变量
init = tf.global_variables_initializer()

# 每个批次的大小，不可能每次只放一张图
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 结果存放在一个布尔型列表中
# argmax返回一维张量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# 定义会话，运行初始化
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(30):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)     # 从batch里面读数据
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})  # 训练

        acc = sess.run(accuracy, feed_dict={                            # 准确度
                       x: mnist.test.images, y: mnist.test.labels})
        print("Iter "+str(epoch)+",Testing Accuracy "+str(acc))



