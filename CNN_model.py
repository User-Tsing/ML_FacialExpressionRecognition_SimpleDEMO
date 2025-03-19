# checked by STAssn
# CNN模型构造
# 简单模型出简单效果，复杂模型对计算机运行内存和GPU效率有较高要求，凑合用吧

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Flatten, Dense, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import PReLU


def CNN1(input_shape=(48, 48, 2), n_classes=7):
    # CNN模型1
    # input
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(32, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # block2
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # block3
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)
    # fc
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


def CNN2(input_shape=(48, 48, 2), n_classes=7):
    # CNN模型2
    # Going deeper with convolutions在输入层后加一层的1*1卷积增加非线性表示

    # input
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(32, (5, 5), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block3
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def CNN3(input_shape=(48, 48, 2), n_classes=7):
    # CNN模型3
    # A Compact Deep Learning Model for Robust Facial Expression Recognition

    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    # block1
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = Conv2D(64, (5, 5), strides=1, padding='same')(x)
    x = PReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


# 8输出CK+适配训练模型
def CNN4(input_shape=(48, 48, 1), n_classes=8):
    # CNN模型4
    # Going deeper with convolutions在输入层后加一层的1*1卷积增加非线性表示

    # input
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(32, (5, 5), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block3
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model


def CNN5(input_shape=(100, 80, 3), n_classes=7):
    # CNN模型5
    # Going deeper with convolutions在输入层后加一层的1*1卷积增加非线性表示

    # input
    input_layer = Input(shape=input_shape)
    # block1
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)
    x = Conv2D(32, (5, 5), strides=1, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block2
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # block3
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    # fc
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model
