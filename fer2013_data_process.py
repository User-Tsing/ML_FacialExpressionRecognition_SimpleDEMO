# directed by STAssn
# fer2013数据集提取数据与预处理

import dlib
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def load_data(csv_file):
    # 读取数据
    data = pd.read_csv(csv_file)

    # 将数据按 'usage' 列进行分组
    train_data = data[data['Usage'] == 'Training']
    public_test_data = data[data['Usage'] == 'PublicTest']
    private_test_data = data[data['Usage'] == 'PrivateTest']

    # 获取训练集的图像和标签
    X_train, y_train = preprocess_data(train_data)

    # 获取公共测试集的图像和标签
    X_public_test, y_public_test = preprocess_data(public_test_data)

    # 获取私有测试集的图像和标签
    X_private_test, y_private_test = preprocess_data(private_test_data)

    return X_train, X_public_test, X_private_test, y_train, y_public_test, y_private_test


# 2. 数据预处理函数
def preprocess_data(data):
    # 提取像素数据并将其转换为numpy数组
    pixels = data['pixels'].tolist()
    X = np.array([np.fromstring(pixel, dtype=int, sep=' ').reshape(48, 48, 1) for pixel in pixels])

    # 数据标准化，将像素值归一化到[0, 1]
    X = X.astype('float32') / 255.0

    # 提取标签并进行编码
    y = data['emotion'].values
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)  # 将标签转化为整数编码
    y = np.array(y)

    return X, y