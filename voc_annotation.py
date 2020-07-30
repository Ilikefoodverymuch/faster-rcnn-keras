import xml.etree.ElementTree as ET
from os import getcwd

# 设置训练、验证、测试集的年份规格和各自名称
sets = [('2007','train'),('2007','val'),('2007','test')]

# 获取当前工作目录
wd = getcwd()

# 设置要识别的目标名称
classes = ["aeroplane", "bicycle", "biat", "bord", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 转换xml文件
def convert_annotation(year,image_id,list_file):
    # 读取xml文件
    in_file = open('VOC%s/Annotations/%s.xml'%(year,image_id))
    # 对xml文件进行解析
    tree = ET.parse(in_file)
    # 返回xml的根元素
    root = tree.getroot()
    # 查找object，如果没有则返回
    if root.find('object') == None:
        return
    # 否则，将文件路径信息写入
    list_file.write('%s/VOC%s/JPEGImages/%s.bmp'%(wd, year, image_id))
    # 遍历xml文件的object对象
    for obj in root.iter('object'):
        # 查找difficult标签（暂时不知道有什么用）
        difficult = obj.find('difficult').text
        # 存储class的名称
        cls = obj.find('name').text
        # 如果该类别不在之前预设的class种类里，或者difficult为1，则跳过
        if cls not in classes or int(difficult) == 1:
            continue
        # 否则，获取该class的序号
        cls_id = classes.index(cls)
        # 获取bonding box标签
        xmlbox = obj.find('bndbox')
        # 获取bonding box标签下的x、y的最大最小值
        b = (int(xmlbox.find('xmin').text),int(xmlbox.find('ymin').text),int(xmlbox.find('xmax').text),int(xmlbox.find('ymax').text))
        # 将刚刚获取的值写入list_file
        list_file.write(" " + ','.join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

# 遍历文件开头设置的训练、验证、测试集年份规格集合
for year,image_set in sets:
    # 读取当前遍历阶段的遍历文件中的图片名称
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    # 打开将要用于最后训练和测试的文件，该文件在根目录，格式类似：2007_test.txt
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    # 遍历当前遍历阶段的遍历文件中的图片
    for image_id in image_ids:
        # 调用convert_annotation方法转换每一个xml文件
        convert_annotation(year, image_id, list_file)
    # 关闭打开的文件
    list_file.close()