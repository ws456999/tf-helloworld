import tensorflow as tf


class Network:
    def __init__(self):
        # 学习速率，一般在 0.00001 - 0.5 之间
        self.learning_rate = 0.001

        # 输入张量 28 * 28 = 784个像素的图片一维向量
        # tf.placeholder(dtype, shape=None, name=None)
        # dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
        # shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
        # name：名称。
        # 返回：Tensor 类型

        self.x = tf.placeholder(tf.float32, [None, 784])

        # 标签值，即图像对应的结果，如果对应数字是8，则对应label是 [0,0,0,0,0,0,0,0,1,0]
        # 这种方式称为 one-hot编码
        # 标签是一个长度为10的一维向量，值最大的下标即图片上写的数字
        self.label = tf.placeholder(tf.float32, [None, 10])

        # 权重，初始化全 0
        '''
        # tf.Variable 看起来像是生成tf自己的专用数据， 比如：
        a1 = tf.Variable(tf.random_normal(shape=[2,3], mean=0, stddev=1), name='a1')
        a2 = tf.Variable(tf.constant(1), name='a2')
        a3 = tf.Variable(tf.ones(shape=[2,3]), name='a3')
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print sess.run(a1)
            print sess.run(a2)
            print sess.run(a3)
        output:
            [[ 0.76599932  0.99722123 -0.89361787]
            [ 0.19991693 -0.16539733  2.16605783]]
            1
            [[ 1.  1.  1.]
            [ 1.  1.  1.]]
        '''
        self.w = tf.Variable(tf.zeros([784, 10]))
        # 偏置 bias， 初始化全 0
        self.b = tf.Variable(tf.zeros([10]))
        # 输出 y = softmax(X * w + b)
        # tf.matmul 表示矩阵的乘法
        # tf.nn.softmax
        # https://my.oschina.net/u/780234/blog/1588827

        self.y = tf.nn.softmax(tf.matmul(self.x, self.w) + self.b)

        # 损失，即交叉熵，最常用的计算标签(label)与输出(y)之间差别的方法
        # 有些地方也叫cost function 用来表示什么时候这个函数达到理想值了 ？ 应该是这样吧， 需要再确定
        #
        # reduce_sum求的是平均值
        '''
        交叉熵 -sum(label * log(y))
        [0, 0, 1] 与 [0.1, 0.3, 0.6]的交叉熵为 -log(0.6) = 0.51
        [0, 0, 1] 与 [0.2, 0.2, 0.6]的交叉熵为 -log(0.6) = 0.51
        [0, 0, 1] 与 [0.1, 0, 0.9]的交叉熵为 -log(0.9) = 0.10
        '''
        self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))

        # 反向传播，采用梯度下降的方法。调整w与b，使得损失(loss)最小
        # loss越小，那么计算出来的y值与 标签(label)值越接近，准确率越高
        # https://www.zhihu.com/question/27239198 介绍反向传播
        self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        # 以下代码验证正确率时使用
        # argmax 返回最大值的下标，最大值的下标即答案
        # 例如 [0,0,0,0.9,0,0.1,0,0,0,0] 代表数字3
        '''
        tf.equal 用法
        A = [[1,3,4,5,6]]
        B = [[1,3,4,3,2]]

        with tf.Session() as sess:
            print(sess.run(tf.equal(A, B)))

        output: [[ True  True  True False False]]
        '''

        '''
        tf.argmax的用法

    　　首先，明确一点，tf.argmax可以认为就是np.argmax。tensorflow使用numpy实现的这个API。

        tf.argmax(vector, 1)：返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。

        A = [[1,3,4,5,6]]
        B = [[1,3,4], [2,4,1]]

        with tf.Session() as sess:
            print(sess.run(tf.argmax(A, 1)))
            print(sess.run(tf.argmax(B, 1)))
        output：
        [4]
        [2 1]
         '''

        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))


        '''
        cast(x, dtype, name=None)
        将x的数据格式转化成dtype.例如，原来x的数据格式是bool，
        那么将其转化成float以后，就能够将其转化成0和1的序列。反之也可以

        a = tf.Variable([1,0,0,1,1])
        b = tf.cast(a,dtype=tf.bool)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        print(sess.run(b))
        #[ True False False  True  True]
         '''
        # predict -> [true, true, true, false, false, true]
        # reduce_mean即求predict的平均数 即 正确个数 / 总数，即正确率
        # tf.reduce_mean表示求平均值
        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))