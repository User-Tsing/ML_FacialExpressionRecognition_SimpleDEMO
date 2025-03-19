# directed by STAssn
# UI界面设计
# 定时器时间是摄像头图像监测，觉得掉帧严重可以减小但请考虑计算机数据处理能力避免不必要的问题

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import QTimer, Qt
import tensorflow as tf
import dlib
from fer import FER


class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        # self.initUI()

        self.setWindowTitle("Facial Expression Recognition 人脸表情识别")  # 标题
        self.setGeometry(500, 100, 750, 860)  # 界面大小

        self.model = tf.keras.models.load_model('trained_model/model_best_3.keras')  # 模型
        self.model_CK = tf.keras.models.load_model('trained_model/model_CK_CNN_7.keras')

        # 放图片的标签
        self.image_label = QLabel(self)  # 标签名称
        self.image_label.setGeometry(50, 50, 650, 500)  # 位置与大小

        # 按钮设置
        self.button_1 = QPushButton("开启摄像头(fer2013)", self)  # 按钮1号
        self.button_1.setGeometry(50, 600, 210, 70)  # 位置与大小
        self.button_1.clicked.connect(self.button_1_click)  # 单击效果：槽函数锁定

        self.button_2 = QPushButton("选择图片(fer2013)", self)  # 按钮2号
        self.button_2.setGeometry(50, 700, 210, 70)  # 位置与大小
        self.button_2.clicked.connect(self.button_2_click)  # 单击效果：槽函数锁定

        self.button_3 = QPushButton("开启摄像头(CK+)", self)  # 按钮3号
        self.button_3.setGeometry(270, 600, 210, 70)  # 位置与大小
        self.button_3.clicked.connect(self.button_3_click)  # 单击效果：槽函数锁定

        self.button_4 = QPushButton("选择图片(CK+)", self)  # 按钮4号
        self.button_4.setGeometry(270, 700, 210, 70)  # 位置与大小
        self.button_4.clicked.connect(self.button_4_click)  # 单击效果：槽函数锁定

        self.button_5 = QPushButton("暂停识别", self)  # 按钮5号
        self.button_5.setGeometry(275, 780, 200, 70)  # 位置与大小
        self.button_5.clicked.connect(self.button_5_click)  # 单击效果：槽函数锁定

        self.button_6 = QPushButton("开启摄像头(API)", self)  # 按钮5号
        self.button_6.setGeometry(490, 600, 210, 70)  # 位置与大小
        self.button_6.clicked.connect(self.button_6_click)  # 单击效果：槽函数锁定

        self.button_7 = QPushButton("选择图片(API)", self)  # 按钮5号
        self.button_7.setGeometry(490, 700, 210, 70)  # 位置与大小
        self.button_7.clicked.connect(self.button_7_click)  # 单击效果：槽函数锁定

        self.my_text_label = QLabel("directed by STAssn", self)  # 标签名称
        self.my_text_label.setGeometry(50, 750, 200, 70)  # 位置与大小

        # 外设配置：定时计数器
        self.timer = QTimer(self)  # 定时计数器
        self.timer.timeout.connect(self.update_frame)  # 溢出“中断回调函数”
        self.timer_CK = QTimer(self)  # 定时计数器
        self.timer_CK.timeout.connect(self.update_frame_CK)  # 溢出“中断回调函数”
        self.timer_API = QTimer(self)  # 定时计数器
        self.timer_API.timeout.connect(self.update_frame_API)  # 溢出“中断回调函数”

        # 外设配置：摄像头
        self.cap = None  # 摄像头，初始化先设定没有

    # 按钮槽函数定义
    def button_1_click(self):
        # 按钮1单击效果函数：设定：打开摄像头并处理
        self.timer_CK.stop()  # 关闭其他定时器
        self.timer_API.stop()
        # self.cap.release()  # 关闭摄像头
        self.open_camera_and_check()  # 调用外设函数

    def button_2_click(self):
        # 按钮2单击效果函数
        self.timer_CK.stop()  # 关闭其他定时器
        self.timer.stop()
        self.timer_API.stop()
        # self.cap.release()  # 关闭摄像头
        self.choose_photo()  # 选取图片处理

    def button_3_click(self):
        # 按钮3单击效果函数：设定：打开摄像头并处理
        self.timer.stop()  # 关闭其他定时器
        self.timer_API.stop()
        # self.cap.release()  # 关闭摄像头
        self.open_camera_and_check_CK()  # 调用外设函数

    def button_4_click(self):
        # 按钮4单击效果函数
        self.timer_CK.stop()  # 关闭其他定时器
        self.timer.stop()
        self.timer_API.stop()
        # self.cap.release()  # 关闭摄像头
        self.choose_photo_CK()  # 选取图片处理

    def button_5_click(self):
        # 按钮5单击效果函数
        self.timer_CK.stop()  # 关闭其他定时器
        self.timer.stop()
        self.timer_API.stop()
        # self.cap.release()  # 关闭摄像头

    def button_6_click(self):
        # 按钮6单击效果函数
        self.timer_CK.stop()  # 关闭其他定时器
        self.timer.stop()
        # self.cap.release()  # 关闭摄像头
        self.open_camera_and_check_API()  # 选取图片处理

    def button_7_click(self):
        # 按钮7单击效果函数
        self.timer_CK.stop()  # 关闭其他定时器
        self.timer.stop()
        self.timer_API.stop()
        # self.cap.release()  # 关闭摄像头
        self.choose_photo_API()  # 选取图片处理

    # 其他协调配置函数
    def open_camera_and_check(self):  # 外设函数：打开摄像头
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        if not self.cap.isOpened():
            print("Failed to open camera")  # 设备没有能用的摄像头（那没办法）
            return
        self.timer.start(30)  # 开启定时计数器，设定20ms中断一次

    def update_frame(self):  # 定时计数器中断函数，本质也是槽函数
        ret, frame = self.cap.read()  # 读取摄像头内容
        if not ret:  # ret是布尔量，看有没有摄像头内容，frame是摄像头拍到的内容
            print("Failed to capture image")
            return
        self.frame_process(frame=frame)  # 调用函数：处理帧

    # 处理帧
    def frame_process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR三色图转化为灰度图

        # 人脸检测，有OpenCV的Haar/DNN、Dlib的HOG、Mediapipe
        # dlib
        # detector = dlib.get_frontal_face_detector()  # 基于dlib的人脸检测模型
        # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 关键点检测
        # faces = detector(gray, 1)

        # Haar
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # cv2的haar级联分类器
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            print("No face detected")
            return

        # 处理检测到的人脸
        # 此处适配dlib
        # for face in faces:
        #     x, y, w, h = (face.left(), face.top(), face.right(), face.bottom())
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # OpenCV画框，暂定
        #     landmarks = predictor(gray, face)

        # 此处适配haar
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # OpenCV画框，暂定
            face_get = frame[y:y+h, x:x+w]  # 获取原图对应人脸位置的图像

            # 预处理：变换灰度，调整大小（无视变形？），准备进模型
            face_gray = cv2.cvtColor(face_get, cv2.COLOR_BGR2GRAY)  # 再度变换灰度图
            face_resized = cv2.resize(face_gray, (48, 48))  # 拉伸图像到48*48（有问题：无视变形）
            face_resized = face_resized.astype('float32') / 255  # 图像灰度级归一化
            face_resized = np.reshape(face_resized, (1, 48, 48, 1))  # 适配模型输入

            # 使用既有模型预测表情
            predictions = self.model.predict(face_resized)
            max_index = np.argmax(predictions[0])  # 找到最大概率的表情

            # 获得当前表情
            labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']  # 检索表
            emoji = labels[max_index]  # 得到的是：模型认为概率最大的表情（不保真）

            # 在人脸框上方显示表情预测
            cv2.putText(frame, emoji, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 循环外：将BGR图像转换为RGB，并显示在PyQt界面上
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR到RGB
        h, w, ch = rgb_image.shape  # 图片长、宽、通道数
        bytes_per_line = ch * w  # 每行字节数：像素数*通道数
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    # 选取照片
    def choose_photo(self):
        options = QFileDialog.Options()  # 图片选取栏
        file_path, _ = QFileDialog.getOpenFileName(self, "表情识别：请选择图片", "", "所有文件 (*);;图像文件 (*.png;*.jpg;*.jpeg;*.bmp)", options=options)

        if file_path:  # 如果有路径
            image = cv2.imread(file_path)  # 打开图片就是做
            self.frame_process(frame=image)  # 调用函数：处理图片

    # CK部分
    def open_camera_and_check_CK(self):  # 外设函数：打开摄像头
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        if not self.cap.isOpened():
            print("Failed to open camera")  # 设备没有能用的摄像头（那没办法）
            return
        self.timer_CK.start(30)  # 开启定时计数器，设定20ms中断一次

    def update_frame_CK(self):  # 定时计数器中断函数，本质也是槽函数
        ret, frame = self.cap.read()  # 读取摄像头内容
        if not ret:  # ret是布尔量，看有没有摄像头内容，frame是摄像头拍到的内容
            print("Failed to capture image")
            return
        self.frame_process_CK(frame=frame)  # 调用函数：处理帧

    def frame_process_CK(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR三色图转化为灰度图

        # 人脸检测，有OpenCV的Haar/DNN、Dlib的HOG、Mediapipe
        # dlib
        # detector = dlib.get_frontal_face_detector()  # 基于dlib的人脸检测模型
        # predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 关键点检测
        # faces = detector(gray, 1)

        # Haar
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # cv2的haar级联分类器
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            print("No face detected")
            return

        # 处理检测到的人脸
        # 此处适配dlib
        # for face in faces:
        #     x, y, w, h = (face.left(), face.top(), face.right(), face.bottom())
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # OpenCV画框，暂定
        #     landmarks = predictor(gray, face)

        # 此处适配haar
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # OpenCV画框，暂定
            face_get = frame[y:y+h, x:x+w]  # 获取原图对应人脸位置的图像

            # 预处理：变换灰度，调整大小（无视变形？），准备进模型
            face_gray = cv2.cvtColor(face_get, cv2.COLOR_BGR2GRAY)  # 再度变换灰度图
            face_resized = cv2.resize(face_gray, (48, 48))  # 拉伸图像到48*48（有问题：无视变形）
            face_resized = face_resized.astype('float32') / 255  # 图像灰度级归一化
            face_resized = np.reshape(face_resized, (1, 48, 48, 1))  # 适配模型输入

            # 使用既有模型预测表情
            predictions = self.model_CK.predict(face_resized)
            max_index = np.argmax(predictions[0])  # 找到最大概率的表情

            # 获得当前表情
            labels = ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']  # CK+检索表
            emoji = labels[max_index]  # 得到的是：模型认为概率最大的表情（不保真）

            # 在人脸框上方显示表情预测
            cv2.putText(frame, emoji, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 循环外：将BGR图像转换为RGB，并显示在PyQt界面上
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR到RGB
        h, w, ch = rgb_image.shape  # 图片长、宽、通道数
        bytes_per_line = ch * w  # 每行字节数：像素数*通道数
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def choose_photo_CK(self):
        options = QFileDialog.Options()  # 图片选取栏
        file_path, _ = QFileDialog.getOpenFileName(self, "表情识别：请选择图片", "", "所有文件 (*);;图像文件 (*.png;*.jpg;*.jpeg;*.bmp)", options=options)

        if file_path:  # 如果有路径
            image = cv2.imread(file_path)  # 打开图片就是做
            self.frame_process_CK(frame=image)  # 调用函数：处理图片

    # API部分
    def open_camera_and_check_API(self):  # 外设函数：打开摄像头
        self.cap = cv2.VideoCapture(0)  # 打开摄像头
        if not self.cap.isOpened():
            print("Failed to open camera")  # 设备没有能用的摄像头（那没办法）
            return
        self.timer_API.start(30)  # 开启定时计数器，设定20ms中断一次

    def update_frame_API(self):  # 定时计数器中断函数，本质也是槽函数
        ret, frame = self.cap.read()  # 读取摄像头内容
        if not ret:  # ret是布尔量，看有没有摄像头内容，frame是摄像头拍到的内容
            print("Failed to capture image")
            return
        self.frame_process_API(frame=frame)  # 调用函数：处理帧

    def frame_process_API(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # BGR三色图转化为灰度图
        predictor = FER()  # 人脸识别模型API

        # 人脸检测，有OpenCV的Haar/DNN、Dlib的HOG、Mediapipe
        # Haar
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # cv2的haar级联分类器
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            print("No face detected")
            return

        # 处理检测到的人脸
        # 此处适配dlib
        # for face in faces:
        #     x, y, w, h = (face.left(), face.top(), face.right(), face.bottom())
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # OpenCV画框，暂定
        #     landmarks = predictor(gray, face)

        # 此处适配haar
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # OpenCV画框，暂定
            face_get = frame[y:y+h, x:x+w]  # 获取原图对应人脸位置的图像

            # 预处理：变换灰度，调整大小（无视变形？），准备进模型
            face_rgb = cv2.cvtColor(face_get, cv2.COLOR_BGR2RGB)  # 再度变换为RGB图

            # 使用既有模型预测表情
            emotion, score = predictor.top_emotion(face_rgb)  # 外部API函数预制模型预测表情

            # 获得当前表情
            emoji = emotion  # 得到的是：模型认为概率最大的表情（不保真）

            # 在人脸框上方显示表情预测
            cv2.putText(frame, emoji, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 循环外：将BGR图像转换为RGB，并显示在PyQt界面上
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR到RGB
        h, w, ch = rgb_image.shape  # 图片长、宽、通道数
        bytes_per_line = ch * w  # 每行字节数：像素数*通道数
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def choose_photo_API(self):
        options = QFileDialog.Options()  # 图片选取栏
        file_path, _ = QFileDialog.getOpenFileName(self, "表情识别：请选择图片", "", "所有文件 (*);;图像文件 (*.png;*.jpg;*.jpeg;*.bmp)", options=options)

        if file_path:  # 如果有路径
            image = cv2.imread(file_path)  # 打开图片就是做
            self.frame_process_API(frame=image)  # 调用函数：处理图片


    # 特别的：画图
    def paintEvent(self, event):
        # 创建 QPainter 对象
        painter = QPainter(self)

        # 设置矩形的颜色
        painter.setBrush(QColor(255, 255, 255))  # 白色填充
        painter.setPen(QColor(0, 0, 0))  # 黑色边框

        # 绘制矩形
        # 使用 QPainter 绘制矩形，x, y 是矩形的左上角坐标，w, h 是宽度和高度
        painter.drawRect(50, 50, 650, 500)  # 矩形坐标 (50, 50)，宽度650，高度500

        painter.end()



# 类外定义：人脸图片缩放不变形
def resize_face(face_gray):
    # 假设 face_gray 是原始灰度图像
    height, width = face_gray.shape

    # 计算保持宽高比的目标尺寸
    if height > width:
        # 如果高度大于宽度，那么按照高度来缩放
        new_width = int(48 * width / height)
        new_height = 48
    else:
        # 如果宽度大于高度，那么按照宽度来缩放
        new_width = 48
        new_height = int(48 * height / width)

    # 进行缩放，保持宽高比
    face_resized = cv2.resize(face_gray, (new_width, new_height))

    # 计算需要填充的空白区域
    top = (48 - new_height) // 2
    bottom = 48 - new_height - top
    left = (48 - new_width) // 2
    right = 48 - new_width - left

    # 填充空白区域，使得最终图像为 48x48
    face_resized = cv2.copyMakeBorder(face_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

    # 现在 face_resized 是 48x48 图像，且保持了原图的宽高比
    return  face_resized
