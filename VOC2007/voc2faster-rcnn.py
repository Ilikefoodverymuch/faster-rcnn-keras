import os
import random

# 该文件下的设置将训练集占训练验证集比率设置为1，即没有设置验证集

# 获取标注xml文件路径
xmlfilepath = r'./VOC2007/Annotations'
# 设置记录训练集、测试集图片序号的存储路径
saveBasePath = r'./VOC2007/ImageSets/Main'

# 设置训练验证集的比率（占总图片）
trainval_percent=0.9
# 设置训练用图片占训练验证集的比率（包含训练和测试）
train_percent=1.0

# 临时记录xml文件名列表
temp_xml = os.listdir(xmlfilepath)
# 保存xml文件名列表
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml")
        total_xml.append(xml)

# 记录xml文件的个数
num = len(total_xml)
# 设置一个和xml文件个数长度相等的list
list = range(num)
# tv用于计算训练验证集图片的个数
tv = int(num*trainval_percent)
# tr用于计算训练集图片的个数
tr = int(tv*train_percent)
# 从总的训练验证集中随机抽取tv个图片用于训练验证
trainval = random.sample(list,tv)
# 从总的训练验证用图片中随机抽取tr个图片用于训练
train = random.sample(trainval,tr)

# 输出训练验证集和训练集的大小
print("train and val size",tv)
print("traub size",tr)

# 打开存储训练验证集图片名的文件
ftrainval = open(os.path.join(saveBasePath,'trainval.txt'),'w')
# 打开存储测试集图片名的文件
ftest = open(os.path.join(saveBasePath,'test.txt'),'w')
# 打开存储训练集图片名的文件
ftrain = open(os.path.join(saveBasePath,'train.txt'),'w')
# 打开存储验证集图片名的文件
fval = open(os.path.join(saveBasePath,'val.txt'),'w')

# 循环xml文件个数次
for i in list:
    # 获取每一个xml文件的名字
    name = total_xml[i][:-4] + "\n"
    # 如果i在trainval里
    if i in trainval:
        # 写入trainval文件
        ftrainval.write(name)
        # 如果i在train里
        if i in train:
            # 写入train文件
            ftrain.write(name)
        # 否则
        else:
            # 写入val文件
            fval.write(name)
    # 否则
    else:
        # 写入测试文件
        ftest.write(name)

# 关闭打开的四个文件
ftrainval.close()
ftrain.close()  
fval.close()  
ftest .close()
