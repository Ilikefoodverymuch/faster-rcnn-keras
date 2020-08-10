from keras.engine.topology import Layer
import keras.backend as K

# 如果keras使用tensorflow作为后端
if K.backend() == "tensorflow":
    import tensorflow as tf

# 将Layer作为RoiPoolingConv的父类
class RoiPoolingConv(Layer):
    '''
    ROI pooling层
    空间金字塔池化
    参数：
        pool_size: int
            要使用的池化层的大小。pool_size = 7 将会产生 7x7 的池化
        num_rois：将使用的 ROI 数目
    输入规格：
        一个张量的列表 [X_img,X_roi] ：
        X_img：
            如果使用Theano作为后端，则X_img的张量形式如下：
            `(1, channels, rows, cols)` if dim_ordering='th'
            如果使用Tensorflow作为后端，则X_img的张量形式如下：
             `(1, rows, cols, channels)` if dim_ordering='tf'.
        X_roi:
            `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    输出规格：
        三维张量
         `(1, num_rois, channels, pool_size, pool_size)`
    '''

    # 初始化各项参数，包括池化大小、rois数量，可选参数
    def __init__(self,pool_size,num_rois,**kwargs):
        # 设定输入格式的规格来自于tf还是th
        self.dim_ordering = K.image_dim_ordering()
        # 限制输入格式的规格必须来自于tf或者th
        assert self.dim_ordering in {'tf','th'}, 'dim_ordering must be in {tf, th}'

        # 设定池化大小
        self.pool_size = pool_size
        # 设定roi数量
        self.num_rois = num_rois

        # 将多余的可选参数传入Layer
        super(RoiPoolingConv,self).__init__(**kwargs)

    # 设定build方法（暂时不懂）
    def build(self,input_shape):
        self.nb_channels = input_shape[0][3]

    # 设定compute_output_shape方法，输入为input_shape向量，输出依次为None，rois数量，池化大小，池化大小，nb_channels
    def compute_output_shape(self,input_shape):
        # 返回
        return None,self.num_rois,self.pool_size,self.pool_size,self.nb_channels
    
    # 
    def call(self,x,mask=None):
        # 限制x的长度为2
        assert(len(x)==2)
        # 共享特征层
        img = x[0]
        # 建议框
        rois = x[1]
        # 输出
        outputs = []

        # 遍历ROI
        for roi_idx in range(self.num_rois):
            #左上角的x,y坐标
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            #建议框的宽和高
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            # 将各项参数转换为int32，之前为张量
            x = K.cast(x,'int32')
            y = K.cast(y,'int32')
            w = K.cast(w,'int32')
            h = K.cast(h,'int32')

            # 将输入的原始图像池化到指定大小：(self.pool_size,self.pool_size)
            rs = tf.image.resize_images(img[:,y:y+h,x:x+w,:],(self.pool_size,self.pool_size))
            # 将本次遍历的ROI添加到output
            outputs.append(rs)

        # 将output的各项从第一个维度开始拼接
        final_output = K.concatenate(outputs,axis=0)
        # 将输出reshape成(1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)形状
        final_output = K.reshape(final_output,(1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        # 进行维度交换，这里没有顺序交换,可以注释掉
        # final_output = K.permute_dimensions(final_output,(0, 1, 2, 3, 4))

        return final_output


