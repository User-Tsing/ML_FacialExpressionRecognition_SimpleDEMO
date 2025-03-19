# directed by STAssn
# CK+数据集提取数据与预处理

import dlib
import numpy as np
import pandas as pd
import os
import cv2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical


# CK+数据集训练模型
def load_data_CK(data_path):
    # 思路：提取照片然后人脸检测，直接把检测到的人脸大头照拿去训练模型
    # 定义情感类别
    emotions = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

    # 图像尺寸
    img_size = (48, 48)

    # cv2的haar级联分类器
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # 提取数据
    images = []
    labels = []
    for idx, emotion in enumerate(emotions):
        emotion_folder = os.path.join(data_path, emotion)
        for filename in os.listdir(emotion_folder):
            if filename.endswith('.png'):
                # 读取图像
                img = cv2.imread(os.path.join(emotion_folder, filename), cv2.IMREAD_GRAYSCALE)
                # 转灰度图（已经是灰度图，不转）
                # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # 提取人脸
                faces = face_cascade.detectMultiScale(img, 1.1, 4)
                # 人脸处理
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]  # 只看第一张人脸
                    face_get = img[y:y + h, x:x + w]  # 获取原图对应人脸位置的图像
                    # face_gray = cv2.cvtColor(face_get, cv2.COLOR_BGR2GRAY)  # 再度变换灰度图
                    face_resized = cv2.resize(face_get, img_size)  # 拉伸图像到48*48
                    face_resized = face_resized.astype('float32') / 255  # 图像灰度级归一化
                    images.append(face_resized)  # 图像信息：只有大头照
                    labels.append(idx)  # 标签信息

    images = np.array(images)
    labels = np.array(labels)
    images = np.expand_dims(images, axis=-1)  # 为CNN添加通道维度 (H, W, 1)
    labels = to_categorical(labels, num_classes=len(emotions))  # 转换为独热编码

    return images, labels

