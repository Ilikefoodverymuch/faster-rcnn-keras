#-------------------------------------------------------------#
#   ResNet50的网络部分
#-------------------------------------------------------------#
from __future__ import print_function

import numpy as np
from keras import layers

from keras.layers import Input
from keras.layers import Dense,Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,TimeDistributed,Add
from keras.layers import Activation,Flatten
from keras.models import Model

from keras.preprocessing import image
import keras.backend as K
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers
from keras import backend as K

# 设定批归一化
class BatchNormalization(Layer):

    # 初始化参数：beta_init：BN中的β，gamma_init：BN中的γ，二者均不采用正则化
    def __init__(self, epsilon=1e-3, axis=-1,
                weights=None, beta_init='zero', gamma_init='one',
                gamma_regularizer=None, beta_regularizer=None, **kwargs):
        
        # 是否对0值进行过滤，主要用于对padding的0值进行忽略
        self.supports_masking = True
        # 对BN中的β进行初始化，初始值为0
        self.beta_init = initializers.get(beta_init)
        # 对BN中的γ进行初始化，初始值为1
        self.gamma_init = initializers.get(gamma_init)
        # 用以防止除0，设置为0.001
        self.epsilon = epsilon
        # 设定计算轴
        self.axis = axis
        # 不采用γ正则化
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        # 不采用β正则化
        self.beta_regularizer = regularizers.get(beta_regularizer)
        # BN的初始权值
        self.initial_weights = weights
        # 继承并传入给Layer
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # 指定输入的ndim,dtype和形状
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        # 指定γ并初始化，trainable=False不可训练
        self.gamma = self.add_weight(shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name),
                                     trainable=False)
        # 指定β并初始化，trainable=False不可训练
        self.beta = self.add_weight(shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name),
                                    trainable=False)
        # 设定平均值并初始化，该初值跟随训练过程更新，但设定了trainable=False不可训练，即采用固定值
        self.running_mean = self.add_weight(shape, initializer='zero',
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)
        # 设定标准差并初始化，该初值跟随训练过程更新，但设定了trainable=False不可训练，即采用固定值
        self.running_std = self.add_weight(shape, initializer='one',
                                           name='{}_running_std'.format(self.name),
                                           trainable=False)

        # 是否初始化权重
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):

        assert self.built, 'Layer must be built before being called'
        # 设定输入规格
        input_shape = K.int_shape(x)

        # 设置channel轴
        reduction_axes = list(range(len(input_shape)))
        # 删除末位以对齐
        del reduction_axes[self.axis]
        # 创建一个与input_shape相同长度的广播数组
        broadcast_shape = [1] * len(input_shape)
        # 修改广播数组的最后维度与input_shape相同
        broadcast_shape[self.axis] = input_shape[self.axis]

        # 如果降维后与x的维度相同
        if sorted(reduction_axes) == range(K.ndim(x))[:-1]:
            # 则不需要对数组进行广播操作
            x_normed = K.batch_normalization(
                x, self.running_mean, self.running_std,
                self.beta, self.gamma,
                epsilon=self.epsilon)
        else:
            # 否则进行广播以对其BN中各项参数（均值、方差、偏移参数）
            broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
            broadcast_running_std = K.reshape(self.running_std, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            x_normed = K.batch_normalization(
                x, broadcast_running_mean, broadcast_running_std,
                broadcast_beta, broadcast_gamma,
                epsilon=self.epsilon)

        # 返回BN层
        return x_normed

    def get_config(self):
        # 基本配置：包含一些初始值和是否对参数采用正则化处理
        config = {
            'epslilon':self.epsilon,
            'axis':self.axis,
            'gamma_regularizer':self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
            'beta_regularizer':self.beta_regularizer.get_config() if self.beta_regularizer else None
        }
        base_config = super(BatchNormalization,self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block):
    # bottleneck瓶颈结构,包含三层，每一层的卷积核数量
    filters1, filters2, filters3 = filters

    # 设置卷积和BN层名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 重复三次BN操作，先卷积，再BN，再激活
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    # 恒等映射，将输入和x相加
    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    
    # 设置卷积核的个数
    filters1, filters2, filters3 = filters

    # 设置卷积和BN层名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # 三次卷积，先卷积，再BN，再激活
    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)
    
    # 残差边的卷积
    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)
    
    # 和残差边的相加
    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNet50(inputs):

    # 记录输入
    img_input = inputs

    # 先对输入进行padding，有三次步长为2的卷积，第一次所用卷积核为(7,7),所以采用(3,3)padding,网络总共将输入的宽高压缩4次
    x = ZeroPadding2D((3, 3))(img_input)
    # 普通卷积，BN，激活操作，进行特征图压缩 -1
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)
    # 最大池化，注意此时采用了same padding，进行特征图压缩 -2
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
 
    # 一个完整的stage，该stage未进行下采样
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 一个完整的stage，该stage进行了下采样，进行特征图压缩 -3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 一个完整的stage，该stage进行了下采样，进行特征图压缩 -4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    return x

def identity_block_td(input_tensor, kernel_size, filters, stage, block, trainable=True):

    # 与上面相同，获取卷积核的数量
    nb_filter1, nb_filter2, nb_filter3 = filters

    # 判断tensor的格式
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # 设定名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 用以处理多输入的TimeDistributed
    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), trainable=trainable, kernel_initializer='normal',padding='same'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2c')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    # 恒等映射
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block_td(input_tensor, kernel_size, filters, stage, block, input_shape, strides=(2, 2), trainable=True):

    # 与上面相同，获取卷积核的数量
    nb_filter1, nb_filter2, nb_filter3 = filters

    # 判断tensor的格式
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # 设定名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 用以处理多输入的TimeDistributed
    x = TimeDistributed(Conv2D(nb_filter1, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), input_shape=input_shape, name=conv_name_base + '2a')(input_tensor)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '2b')(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(nb_filter3, (1, 1), kernel_initializer='normal'), name=conv_name_base + '2c', trainable=trainable)(x)
    x = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '2c')(x)

    # 跳跃连接
    shortcut = TimeDistributed(Conv2D(nb_filter3, (1, 1), strides=strides, trainable=trainable, kernel_initializer='normal'), name=conv_name_base + '1')(input_tensor)
    shortcut = TimeDistributed(BatchNormalization(axis=bn_axis), name=bn_name_base + '1')(shortcut)

    # 恒等映射
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def classifier_layers(x, input_shape, trainable=False):
    
    # 采用一个卷积下采样层和两个恒等映射，第五次压缩宽高
    x = conv_block_td(x, 3, [512, 512, 2048], stage=5, block='a', input_shape=input_shape, strides=(2, 2), trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='b', trainable=trainable)
    x = identity_block_td(x, 3, [512, 512, 2048], stage=5, block='c', trainable=trainable)

    # 进行平均池化，进一步压缩特征
    x = TimeDistributed(AveragePooling2D((7, 7)), name='avg_pool')(x)

    return x