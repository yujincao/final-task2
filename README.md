# 在CIFAR-100数据集上比较基于Transformer和CNN的图像分类模型

安装好python环境

执行 pip install -r requirements.txt 安装依赖

训练可以再gpu或者cpu环境下

运行 cnn_train.py
在CIFAR-100数据集上训练CNN模型
评测结束保存loss和acc图像

运行 cnn_cutmix_train.py
在CIFAR-100数据集上使用cutmix数据增强后训练CNN模型
评测结束保存loss和acc图像

运行 cnn_deep_train.py
在CIFAR-100数据集上训练加深卷积层后的CNN模型
评测结束保存loss和acc图像

运行 transformer_train.py
在CIFAR-100数据集上训练transformer模型
评测结束保存loss和acc图像

运行 transformer_cutmix_train.py
在CIFAR-100数据集上使用cutmix数据增强后训练transformer模型
评测结束保存loss和acc图像

运行 transformer_deep_train.py

在CIFAR-100数据集上训练加深模型后的transformer模型

评测结束保存loss和acc图像



数据权重地址链接：https://pan.baidu.com/s/1Nge2ESRDiI7uop6d_Uf-wQ
提取码：lu02
