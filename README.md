## tensorflow helloworld

> tensorflow入门-mnist手写数字识别 难度等同于正常世界中的hello，world
>
> 之前面试的那家金融公司就觉得自己的机器学习很吊，我希望我能先学一些数学知识再深入一些来写这个东西
>
> 我在这里写出来，是为了方便自己记忆，如果能帮到别人的话，那当然也是最好的

MNIST是一个入门级的计算机视觉数据集，它包含各种手写数字图片。在机器学习中的地位相当于Python入门的打印Hello World。官网是[THE MNIST DATABASE of handwritten digits](http://yann.lecun.com/exdb/mnist/)

[inspire](https://geektutu.com/post/tensorflow-mnist-simplest.html)

### 关键词
- 数据集
- 独热编号
- 图片 / 标签
- loss
- 回归模型
- 学习速率
- 激活函数


### 理解

> 矩阵绝对是机器学习所有地方都会用到的东西 [理解矩阵乘法
](http://www.ruanyifeng.com/blog/2015/09/matrix-multiplication.html)

1. 定义好一些初始化参数
`learning_rate = 0.001 # 学习速率`

2. 设定

```py
# [None, 3]表示列是3，行不定 其实这里就是定义了一个一维数组
# 这些数据会在sess.run的时候通过feed_dict传进来的
self.x = tf.placeholder(tf.float32, [None, 784])
self.label = tf.placeholder(tf.float32, [None, 10])

# 设定因变量 w 跟 b
# Variable（）构造函数需要变量的初始值，它可以是任何类型和形状的Tensor（张量）。 初始值定义变量的类型和形状。 施工后，变量的类型和形状是固定的。 该值可以使用其中一种赋值方式进行更改。
self.w = tf.Variable(tf.zeros([784, 10]))
# 偏置 bias， 初始化全 0
self.b = tf.Variable(tf.zeros([10]))
```

3. 开始训练数据

batch_size每次训练集大小
tran_step训练次数

然后就可以从既定的数据集里面拿数据了
每次都拿出一个训练集大小的图片 => x, label 就是他们的集合（x表示矩阵，label表示他的实际值）

```py
# 初步了解，loss表示一个批次中矩阵点阵与输入的不同平均个数，一共是28 * 28 个点阵，上面初始化的时候定义了784个w，意味着这个一个函数聚合，w1 + w2 + w3 + ...这样的东西
_, loss = self.sess.run(
  [self.net.train, self.net.loss],
  feed_dict={self.net.x: x,self.net.label: label}
)

这里就运行了train 跟 loss 函数，顺序不太确定，但是这不是需要关心的地方
```

4. 开始退回去开train 跟loss

不过在机器学习中，一般是采用成本函数（cost function），然后，训练目标就是通过调整每一个权值Wij来使得cost达到最小。cost函数也可以看成是由所有待求权值Wij为自变量的复合函数，而且基本上是非凸的，即含有许多局部最小值。但实际中发现，采用我们常用的梯度下降法就可以有效的求解最小化cost函数的问题
Cost | loss 其实就是我们训练的目标值，为了让这个值变小，找到函数的凹处
[[如何直观地解释 back propagation 算法？ - 知乎](https://www.zhihu.com/question/27239198)]
```py
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
```


### 问题

- 梯度下降这个函数是怎么影响w / b两个变量的
- 初始化把所有的w 跟 b都设置成0，真的合适吗
- Softmax 其实就是激励函数，为的是一是放大效果，而是梯度下降时需要一个可导的函数，而且这个激励函数，之后会有把二维函数，拉升成N维函数的作用，贼tm吊，具体是怎么激励的
