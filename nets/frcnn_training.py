from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
import keras
import tensorflow as tf
import numpy as np
from random import shuffle
import random
from PIL import Image
from keras.objectives import categorical_crossentropy
from keras.utils.data_utils import get_file
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from utils.anchors import get_anchors
import time 

# 产生随机数
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

def cls_loss(ratio=3):
    def _cls_loss(y_true, y_pred):
        # y_true [batch_size, num_anchor, num_classes+1]
        # y_pred [batch_size, num_anchor, num_classes]
        # 获取标注的数据
        labels         = y_true
        # 单独取出标注的状态，-1 是需要忽略的, 0 是背景, 1 是存在目标
        anchor_state   = y_true[:,:,-1] 
        # 获取分类结果
        classification = y_pred

        # 找出存在目标的先验框，1 是存在目标
        indices_for_object        = tf.where(keras.backend.equal(anchor_state, 1))
        labels_for_object         = tf.gather_nd(labels, indices_for_object)
        classification_for_object = tf.gather_nd(classification, indices_for_object)

        # 计算类别损失
        cls_loss_for_object = keras.backend.binary_crossentropy(labels_for_object, classification_for_object)

        # 找出实际上为背景的先验框
        indices_for_back        = tf.where(keras.backend.equal(anchor_state, 0))
        labels_for_back         = tf.gather_nd(labels, indices_for_back)
        classification_for_back = tf.gather_nd(classification, indices_for_back)

        # 计算每一个先验框应该有的权重
        cls_loss_for_back = keras.backend.binary_crossentropy(labels_for_back, classification_for_back)

        # 标准化，实际上是正样本的数量
        normalizer_pos = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer_pos = keras.backend.cast(keras.backend.shape(normalizer_pos)[0], keras.backend.floatx())
        normalizer_pos = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_pos)

        # 标准化，实际上是负样本的数量
        normalizer_neg = tf.where(keras.backend.equal(anchor_state, 0))
        normalizer_neg = keras.backend.cast(keras.backend.shape(normalizer_neg)[0], keras.backend.floatx())
        normalizer_neg = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer_neg)
        
        # 将所获得的loss除上正样本的数量
        cls_loss_for_object = keras.backend.sum(cls_loss_for_object)/normalizer_pos
        cls_loss_for_back = ratio*keras.backend.sum(cls_loss_for_back)/normalizer_neg

        # 总的loss
        loss = cls_loss_for_object + cls_loss_for_back

        return loss
    return _cls_loss

def smooth_l1(sigma=1.0):
    # 
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # y_true [batch_size, num_anchor, 4+1]
        # y_pred [batch_size, num_anchor, 4]
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # 找到正样本
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # 计算 smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # 
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        loss = keras.backend.sum(regression_loss) / normalizer

        return loss

    return _smooth_l1

def class_loss_regr(num_classes):
    # 计算回归损失
    epsilon = 1e-4
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4*num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        loss = 4*K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])
        return loss
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    # 计算分类损失
    return K.mean(categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))

def get_new_img_size(width, height, img_min_side=600):
    # 计算图片大小
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = int(img_min_side)
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = int(img_min_side)

    return resized_width, resized_height

def get_img_output_length(width, height):
    # 
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [7, 3, 1, 1]
        padding = [3,1,0,0]
        stride = 2
        for i in range(4):
            # input_length = (input_length - filter_size + stride) // stride
            input_length = (input_length+2*padding[i]-filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width), get_output_length(height) 