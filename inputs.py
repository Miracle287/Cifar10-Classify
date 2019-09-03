# -*- coding:utf-8 -*-

import os
import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import torchvision

# 设定类别总数
NUM_CLASSES = 10
# 设定训练样本数
NUM_PER_EPOCH_FOR_TRAIN = 50000
# 测试样本数
NUM_PER_EPOCH_FOR_TEST = 10000

# 训练数据目录
input_train_dir = "input/train"
# 测试数据目录
input_test_dir = "input/test"
# 图片缓存目录
input_train_image_origin_dir = "input/train/image_origin"
input_test_image_origin_dir = "input/test/image_origin"
# 数据增强的图片目录
input_train_image_distorted_dir = "input/train/image_distorted"
input_test_image_distorted_dir = "input/test/image_distorted"

# 模型结果输出目录
output_dir = "output"

# 类别标签数组
label_name_array = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
label_dict = {
	"airplane"     :0,
	"automobile"   :1,
	"bird"         :2,
	"cat"          :3,
	"deer"         :4,
	"dog"          :5,
	"frog"         :6,
	"horse"        :7,
	"ship"         :8,
	"truck"        :9
}


# 存储pickle到文件, 不能保存tensor变量
def save(file, data):
    with open(file, 'wb') as fw:
        pickle.dump(data, fw, pickle.HIGHEST_PROTOCOL)


# 文件中读取pickle
def load(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo)
    return data


# 使用pickle打开被封装的对象，返回字典数据，包含图片数据data和类别标签
def unpickle(file):
    print("unpickle file: " + os.path.abspath(file))
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict


# 读取一个文件的数据，返回data和label数组
def read_batch(file):
    batch_data = unpickle(file)
    data = batch_data['data']
    label = batch_data['labels']
    label = np.array(label)
    del batch_data
    return data, label


# 读取cifar-10 原始数据
def read_cifar10(file, data_num):
    # 定义一个空类, 继承object类, 读取返回的cifar-10数据
    class Cifar10Record(object):
        pass
    result = Cifar10Record()

    label_bytes = 1    # cifar-10为1，cifar-100为2

    result.height = 32
    result.width = 32
    result.depth = 3    # 图像都为RGB，因此有三层

    image_bytes = result.height * result.width * result.depth    # 每张图的大小
    record_bytes = image_bytes + label_bytes                     # 每个样本包含一个image数据和label数据

    # 数据预处理, 获取tensor对象, tensor数据只能使用tf的函数预处理
    image_data, label_data = read_batch(file)

    result.label = tf.cast(label_data, tf.int32)      # uint8转变成int32数据类型
    image_data = np.reshape(image_data, (data_num, result.depth, result.height, result.width))
    image_data = np.transpose(image_data, (0, 2, 3, 1))

    # 将原始图像输出
    # show_pic(image_data[34], label_data[34])

    result.image = image_data
    result.label = label_data

    return result


# 对数据做one-hot编码
def make_one_hot_labels():
    n_class = 10
    n_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = tf.one_hot(n_labels, n_class, 1, 0)
    with tf.Session() as sess:
        out_label = sess.run(b)
    return out_label


# 处理训练数据
def process_train_data():
    output_folder = input_train_image_origin_dir

    data_len = 10000

    label_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    files = os.listdir("assets")
    files.sort()
    for file_name in files:
        # data_batch的是训练数据
        if file_name.find("data_batch") >= 0:

            print("process file:", file_name)
            result_data = read_cifar10(os.path.join("assets", file_name), data_len)
            # 图片数据预处理
            for step in range(data_len):
                image, label = result_data.image[step], result_data.label[step]

                label_name = label_name_array[result_data.label[step]]
                label_dir = os.path.join(output_folder, label_name)
                if not os.path.exists(label_dir):
                    os.makedirs(label_dir, exist_ok=True)

                # 保存图片到文件
                num = label_num[result_data.label[step]]
                pic_file_name = label_name + "_" + str(num) + ".jpg"
                misc.imsave(os.path.join(label_dir, pic_file_name), image)

                label_num[result_data.label[step]] += 1

                if step % 100 == 0:
                    print("process step: %d" % (step))


def process_train_data_distorted():
    input_folder = input_train_image_origin_dir
    output_folder = input_train_image_distorted_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for root, sub_dirs, files in os.walk(input_folder):
        sub_dirs.sort()
        files.sort()
        # Browse all files
        for filename in files:
            file_path = os.path.join(root, filename)
            label_name = os.path.split(root)[1]

            print("process file: ", filename, label_name)

            # 数据增强处理
            image = Image.open(file_path)
            image = get_data_with_distorted(image)

            label_dir = os.path.join(output_folder, str(label_name))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir, exist_ok=True)

            # 保存到文件
            misc.imsave(os.path.join(label_dir, filename), image)


# 处理测试数据
def process_test_data():
    output_folder = input_test_image_origin_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    label_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    result_data = read_cifar10(os.path.join("assets", "test_batch"), NUM_PER_EPOCH_FOR_TEST)
    for step in range(NUM_PER_EPOCH_FOR_TEST):
        image, label = result_data.image[step], result_data.label[step]

        label_name = label_name_array[result_data.label[step]]
        label_dir = os.path.join(output_folder, label_name)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir, exist_ok=True)

        # 保存图片到文件
        num = label_num[result_data.label[step]]
        pic_file_name = label_name + "_" + str(num) + ".jpg"
        misc.imsave(os.path.join(label_dir, pic_file_name), image)

        label_num[result_data.label[step]] += 1

        if step % 100 == 0:
            print("process step: %d" % (step))


def process_test_data_distorted():
    input_folder = input_test_image_origin_dir
    output_folder = input_test_image_distorted_dir

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for root, sub_dirs, files in os.walk(input_folder):
        sub_dirs.sort()
        files.sort()
        # Browse all files
        for filename in files:
            file_path = os.path.join(root, filename)
            label_name = os.path.split(root)[1]

            print("process file: ", filename, label_name)

            # 不做数据增强处理
            image = Image.open(file_path)
            image = get_data_without_distorted(image)

            label_dir = os.path.join(output_folder, label_name)
            if not os.path.exists(label_dir):
                os.makedirs(label_dir, exist_ok=True)

            # 保存到文件
            misc.imsave(os.path.join(label_dir, filename), image)


# 图片数据预处理
def get_data_with_distorted(image):
    # 将图片裁剪成24*24*3大小
    cropped_image = torchvision.transforms.RandomCrop(24)(image)
    # 随机左右翻转图片
    flip_image = torchvision.transforms.RandomVerticalFlip()(cropped_image)
    # 调整亮度, 对比度
    adjust_color_image = torchvision.transforms.ColorJitter(brightness=1, contrast=1)(flip_image)
    return adjust_color_image


def get_data_without_distorted(image):
    # 不改变深度，只裁剪大小
    cropped_image = torchvision.transforms.RandomCrop(24)(image)
    return cropped_image


# save feature data
def save_image_and_label(input_dir, output_dir):
    _x = []
    _y = []
    one_hot_label_list = make_one_hot_labels()
    for root, sub_dirs, files in os.walk(input_dir):
        sub_dirs.sort()
        files.sort()
        # Browse all files
        for filename in files:
            file_path = os.path.join(root, filename)
            label_name = os.path.split(root)[1]
            label_code = label_dict[label_name]
            if filename == ".DS_Store":
                os.remove(file_path)
            else:
                # print("Load image: ", file_path, "label: ", one_hot_label_list[label_code])
                _x.append(np.asarray(Image.open(file_path), dtype="float32"))
                _y.append(one_hot_label_list[label_code])

    _x, _y = np.array(_x), np.array(_y)
    print(_x.shape, _y.shape)
    save(os.path.join(output_dir, "data.dat"), _x)
    save(os.path.join(output_dir, "label.dat"), _y)
    return _x, _y


# 显示某一张rgb图片(tensor对象需调用eval()方法)
def show_pic(image, label):
    # 显示最后得到的rgb图片
    print(image.shape)
    print(image.dtype)
    plt.imshow(image)

    # 默认图例和其他显示设置，并显示
    plt.title(label=label)
    plt.legend()
    plt.show()


# 程序入口：数据预处理
if __name__ == '__main__':
    process_train_data()
    process_train_data_distorted()
    process_test_data()
    process_test_data_distorted()
    save_image_and_label(input_dir=input_train_image_distorted_dir, output_dir=input_train_dir)
    save_image_and_label(input_dir=input_test_image_distorted_dir, output_dir=input_test_dir)









