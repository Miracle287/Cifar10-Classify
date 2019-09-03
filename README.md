## CifarClassifier

Python版Cifar-10数据集图片分类程序

由于官方给出的Demo使用Cifar-10数据集是二进制的。本项目使用Python版Cifar-10数据集写了一个图片分类的Demo



#### 环境

深度学习框架：TensorFlow、PyTorch

开发环境: PyCharm



#### 使用

在inputs.py中配置相关目录：

```python
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
```



执行inputs.py生成数据集

```python
python inputs.py
```



执行train,py开始训练

```
python train.py
```





