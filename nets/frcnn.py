from nets.resnet import ResNet50,classifier_layers
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model
from nets.RoiPoolingConv import RoiPoolingConv

# 网络中的kernel_initializer和activation都可以自由选择

# 获取RPN网络结果，输入：基础层，anchors的数量
def get_rpn(base_layers,num_anchors):
    # 采用512个卷积核对输入进行3x3卷积
    x = Conv2D(512,(3,3),padding='same',activation='relu',kernel_initializer='normal',name='rpn_conv1')(base_layers)
    # 对上层x的卷积结果再进行卷积，卷积核数量和先验框的数量应保持一致，卷积核大小为1x1
    x_class = Conv2D(num_anchors,(1,1),activation="sigmoid",kernel_initializer="uniform",name="rpn_out_class")(x)
    # 对上层x的卷积结果再进行卷积，卷积核的数量为先验框的数量的4倍，卷积核大小为1x1
    x_regr = Conv2D(num_anchors*4,(1,1),activation="linear",kernel_initializer="zero",name="rpn_out_regress")(x)
    # 对上层x_class的输出进行reshape，其中-1代表该reshape不关心输出的行数，但列数应指定为1
    x_class = Reshape((-1,1),name="classification")(x_class)
    # 对上层x_regr的输出进行reshape，其中-1代表该reshape不关心输出的行数，但列数应指定为4
    x_regr = Reshape((-1,4),name="regression")(x_regr)
    # 返回x_class、x_regr、base_layers
    return [x_class,x_regr,base_layers]

# 获取分类网络结果，输入：基础层、输入rois、rois数量、识别类别数量（默认为voc的20类，但应加上背景类，故为21），是否可以被训练
def get_classifier(base_layers,input_rois,num_rois,nb_classes=21,trainable=False):
    
    pooling_regions = 14
    input_shape = (num_rois,14,14,1024)
    # base_layers -> [38,38,1024] 当输入为[600,600,3] 的图片的时候
    #input_rois -> [None, 4]，None一般每次处理为32个建议框，在config.py中的num.rois中设置，
    out_roi_pool = RoiPoolingConv(pooling_regions,num_rois)([base_layers,input_rois])
    out = classifier_layers(out_roi_pool,input_shape=input_shape,trainable=True)
    out = TimeDistributed(Flatten())(out)
    out_class = TimeDistributed(Dense(nb_classes,activation="softmax",kernel_initializer="zero"),name="dense_class_{}".format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4*(nb_classes-1),activation="linear",kernel_initializer="zero"),name="dense_regress_{}".format(nb_classes))(out)
    return [out_class,out_regr]