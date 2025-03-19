# directed by STAssn
# ed数据集提取数据与预处理

import dlib
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


def load_data_ed(data_path):
    # 思路：提取照片然后人脸检测，直接把检测到的人脸大头照拿去训练模型
    # 定义情感类别
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # 图像尺寸
    img_size = (80, 100)

    # cv2的haar级联分类器
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 提取数据
    images = []
    labels = []
    for idx, emotion in enumerate(emotions):
        emotion_folder = os.path.join(data_path, emotion)
        for filename in os.listdir(emotion_folder):
            if filename.endswith('.jpg'):
                # 读取图像
                img = cv2.imread(os.path.join(emotion_folder, filename))  # 彩图
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_reshape = cv2.resize(img_rgb, img_size)
                img_standard = img_reshape.astype('float32') / 255.0
                images.append(img_standard)  # 图像信息
                labels.append(idx)

    images = np.array(images)
    labels = np.array(labels)
    images = np.expand_dims(images, axis=-1)  # 为CNN添加通道维度 (H, W, 1)
    labels = to_categorical(labels, num_classes=len(emotions))  # 转换为独热编码

    return images, labels
