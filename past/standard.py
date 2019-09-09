#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : 模型构建规范.py
# @Date  : 2019/4/20 0020
# @Contact : 1329778364@qq.com 
# @Author: DeepMan
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.layers import *


class myModel(Model):
    def __init__(self):
        super(myModel, self).__init__()
        """
        初始化我们自己需要 使用到的神经网络层。
        """
        self.rows = None
        self.cols = None
        self.kernel = (3, 3)
        self.init = RandomNormal(stddev=0.01)
        self.model = Sequential()


    # 实例化模型调用的时候 将训练数据传入
    def call(self, x):
        """
        利用自定义的层进行堆叠 构成我们需要的神经网络。
        :param x:
        :return:
        """
        self.model.add(Conv2D(64, kernel_size=self.kernel, activation='relu', padding='same', input_shape=(self.rows, self.cols, 3),
                         kernel_initializer=self.init))
        self.model.add(Conv2D(64, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))
        self.model.add(MaxPooling2D(strides=2))
        self.model.add(Conv2D(128, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))
        self.model.add(Conv2D(128, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))
        self.model.add(MaxPooling2D(strides=2))
        self.model.add(Conv2D(256, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))
        self.model.add(Conv2D(256, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))
        self.model.add(Conv2D(256, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))
        self.model.add(MaxPooling2D(strides=2))
        self.model.add(Conv2D(512, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))
        self.model.add(Conv2D(512, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))
        self.model.add(Conv2D(512, kernel_size=self.kernel, activation='relu', padding='same', kernel_initializer=self.init))

        # Conv2D
        self.model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=self.init, padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=self.init, padding='same'))
        self.model.add(Conv2D(512, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=self.init, padding='same'))
        self.model.add(Conv2D(256, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=self.init, padding='same'))
        self.model.add(Conv2D(128, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=self.init, padding='same'))
        self.model.add(Conv2D(64, (3, 3), activation='relu', dilation_rate=2, kernel_initializer=self.init, padding='same'))
        self.model.add(Conv2D(1, (1, 1), activation='relu', dilation_rate=1, kernel_initializer=self.init, padding='same'))


"""
1 数据准备阶段，将需要使用的数据集准备好
    
    几个常用操作：
    1.1 对数据进行切分 并对长度不满足的进行丢弃。放回的是可迭代的对象 其中每一项表示一个length长度
    chunk = tf.data.Dataset.from_tensor_slices(
    text_as_int).batch(seq_length + 1, drop_remainder=True)
    
    1.2 
    对数据进行切分文训练数据 和 标签
    def spilit_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text
    dataset = chunk.map(spilit_input_target)
    
    1.3 对数据进行shuffle 保证数据的随机性，分布是正态分布：
    BUFFER_SIZE = 1000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(
        BATCH_SIZE, drop_remainder=True)
    
"""



"""
2 模型构建阶段
    2.1 首先需要实例化模型
    model = MyModel(vocab_size, embedding_dim, units)
    
    2.2 定义优化函数 
    optimizer = tf.train.AdadeltaOptimizer()

    2.3 定义损失函数
    def loss_function(real, predict):
        return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=predict)
        
    2.4 对模型进行build 使得模型知道我们输入数据的格式。其中使用tensorshanpe;
    参数包含[BATCH_SIZE, seq_length]，即我们一次输入数据个个数，和每个数据的长度信息
    model.build(tf.TensorShape([BATCH_SIZE, seq_length]))
    
    2.5 输出 模型的基本信息
    model.summary()


"""




""""
3. 训练模型阶段
    3.1 表示训练几个EPOCH:
    for epoch in range(EPOCHS):
        每个epoch 相当于重新对模型进行一次训练 对于RNN模型 需要将model的hidden 
        层初始状态进行Reset才能较好的学习w 和 b
        hidden = model.reset_states()
        
        由于上面数据准备阶段生成的dataset 中还包含了batch id 表示第几个batch
        for (batch, (inp, target)) in enumerate(dataset):

            进行梯度下降进行训练；
            with tf.GradientTape() as tape:
                predict = model(inp)
                loss = loss_function(target, predict)
                
            根据Loss 求针对 可训练参数的梯度，然后使用梯度来进行更新权值和bias 每个batch更新一次网络参数。
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
            
            输出训练工程中的基本信息 利用format：
            print("Epoch {} batch {} Loss {:.4f}".format(
                        epoch + 1, batch, loss))
                        
                或者将训练过程中的基本信息（参数 loss acc batch epoch AUC 等等参数这个我们再训练之前进行讨论规划好）进行存储，以便后面进行可视化。
                
"""





"""
4. 对模型进行存储 
    
    4.1 存储模型的地址
    checkpoint_dir = "./training_checkpoints"
    # 根据路径创建目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 命名 ckpt
    checkpoint_prefix = os.path.join(checkpoint_dir, "ChauffeurNet_{}_{}.ckpt".format(epoch, batch))

        
    4.2 对模型的参数进行存储
    两种方式：
        A.直接使用save:
        model.save_weights(checkpoint_prefix)
    
        B 使用eager模式下
        root = tf.train.Checkpoint(optimizer=optimizer,
                                   model=model,
                                   optimizer_step=tf.train.get_or_create_global_step())
        checkpoint_prefix = os.path.join(checkpoint_dir, "ChauffeurNet_{}.ckpt".format(epoch))
        root.save(checkpoint_prefix)
    
"""



"""
5. 对训练好的模型进行调用 
    1  先使用实例化模型
    model = MyModel(vocab_size, embedding_dim, units)
    
    2  读取模型最新的权重
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    
    3  将模型的输入格式进行初始化
    model.build(tf.TensorShape([1, None]))
    
    4  输入数据进行预测
"""



















