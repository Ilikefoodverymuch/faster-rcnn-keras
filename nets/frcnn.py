from nets.resnet import ResNet50,classifier_layers
from keras.layers import Conv2D,Input,TimeDistributed,Flatten,Dense,Reshape
from keras.models import Model
from nets.RoiPoolingConv import RoiPoolingConv

# 网络中的kernel_initializer和activation都可以自由选择

# 获取RPN网络结果，输入：基础层，anchors的数量
def get_rpn(base_layers,num_anchors):
    # 采用512个卷积核对输入进行3x3卷积，输入为特征提取层，即resnet50的输出
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
    # 设定池化大小
    pooling_regions = 14
    # 设定输入大小（roi数量，池化大小，池化大小，固定长度）
    input_shape = (num_rois,14,14,1024)
    # base_layers -> [38,38,1024] 当输入为[600,600,3] 的图片的时候
    #input_rois -> [None, 4]，None一般每次处理为32个建议框，在config.py中的num.rois中设置，
    # 获取RoiPoolingConv层输出的公共特征层
    out_roi_pool = RoiPoolingConv(pooling_regions,num_rois)([base_layers,input_rois])
    # 获取分类层的输出结果
    out = classifier_layers(out_roi_pool,input_shape=input_shape,trainable=True)
    # 进行flatten操作，准备进行全连接输出分类结果
    out = TimeDistributed(Flatten())(out)
    # 进行全连接输出分类结果
    out_class = TimeDistributed(Dense(nb_classes,activation="softmax",kernel_initializer="zero"),name="dense_class_{}".format(nb_classes))(out)
    # 进行全连接输出调整后的框位置
    out_regr = TimeDistributed(Dense(4*(nb_classes-1),activation="linear",kernel_initializer="zero"),name="dense_regress_{}".format(nb_classes))(out)
    # 返回
    return [out_class,out_regr]

def get_model(config,num_classes):
    # 设定输入
    inputs = Input(shape=(None, None, 3))
    # 设定roi的个数以及他们的位置坐标，None一般为32，可以在config.py中的num.rois中设置
    roi_input = Input(shape=(None, 4))
    # 获取特征提取层的输出
    base_layers = ResNet50(inputs)

    # 一般采用9个anchor，可以在config.py中设置
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    # 获取RPN
    rpn = get_rpn(base_layers, num_anchors)
    # 建立rpn模型
    model_rpn = Model(inputs, rpn[:2])

    # 获取分类模型的输出
    classifier = get_classifier(base_layers, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    # 建立分类模型
    model_classifier = Model([inputs, roi_input], classifier)

    # 同时包含RPN和分类网络的model，用来保存和读取权重信息
    model_all = Model([inputs, roi_input], rpn[:2]+classifier)

    # 返回模型
    return model_rpn,model_classifier,model_all

def get_predict_model(config,num_classes):

    # 设定输入
    inputs = Input(shape=(None, None, 3))
    # 设定roi的个数以及他们的位置坐标，None一般为32，可以在config.py中的num.rois中设置
    roi_input = Input(shape=(None, 4))
    # 输入特征层规格
    feature_map_input = Input(shape=(None,None,1024))

    # 获取特征提取层的输出
    base_layers = ResNet50(inputs)
    # 生成多少个anchor box
    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
    # 获取RPN
    rpn = get_rpn(base_layers, num_anchors)
    # 建立rpn模型
    model_rpn = Model(inputs, rpn)

    # 获取分类模型的输出
    classifier = get_classifier(feature_map_input, roi_input, config.num_rois, nb_classes=num_classes, trainable=True)
    # 分类模型
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    # 返回模型
    return model_rpn,model_classifier_only