import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model,layers,Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation
from tensorflow.keras.applications import vgg16
import numpy as np
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import math
from keras_flops import get_flops
import tensorflow_addons as tfa


def Unet(input_shape=(224,224,3), num_classes=21):
    '''
      [B,H,W,C] = x.shape
    '''
    img_input=Input(shape=input_shape)

    assert img_input.shape[1] % 32 == 0,\
        'The input_height should be divisible by 32'
    assert img_input.shape[2] % 32 == 0,\
        'The input_width should be divisible by 32'

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)
    assert isinstance(model, Model), 'The model should be the instantiation of the keras Model '


    # -------------------------------------------------#
    #              Decoder
    # -------------------------------------------------#
    # 7,7,512 -> 14,14,1024 -> 14,14,512
    P5 = UpSampling2D((2, 2))(model.output)
    P5 = concatenate([model.get_layer(
        name="block4_pool").output, P5], axis=-1)
    P5 = Conv2D(512, (3, 3), padding="same")(P5)
    P5 = BatchNormalization()(P5)

    # 14,14,512 -> 28,28,768 -> 28,28,256
    P4 = UpSampling2D((2, 2))(P5)
    P4 = concatenate([model.get_layer(
        name="block3_pool").output, P4], axis=-1)
    P4 = Conv2D(256, (3, 3), padding="same")(P4)
    P4 = BatchNormalization()(P4)

    # 28,28,256 -> 56,56,384 -> 56,56,128
    P3 = UpSampling2D((2, 2))(P4)
    P3 = concatenate([model.get_layer(
        name="block2_pool").output, P3], axis=-1)
    P3 = Conv2D(128, (3, 3), padding="same")(P3)
    P3 = BatchNormalization()(P3)

    # 56,56,128 -> 112,112,192 -> 112,112,64
    P2 = UpSampling2D((2, 2))(P3)
    P2 = concatenate([model.get_layer(
        name="block1_pool").output, P2], axis=-1)
    P2 = Conv2D(64, (3, 3), padding="same")(P2)
    P2 = BatchNormalization()(P2)

    # UNet网络处理输入时进行了镜面放大2倍，所以最终的输入输出缩小了2倍
    # 此处直接上采样原始大小
    # 112,112,64 -> 224,224,64
    x = UpSampling2D((2, 2))(P2)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    x = Conv2D(num_classes, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('softmax')(x)

    model = Model(inputs=img_input, outputs=x)
    return model


if __name__ == '__main__':
    model= Unet()
    # print(m.get_weights()[2]) # 看看权重改变没，加载vgg权重测试用
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='model_unet.png')
    print(len(model.layers))
    model.summary()
    # flops = get_flops(Unet(), batch_size=1) / (2 * 10 ** 9)
    # print(f"FLOPs:{flops:.1f}G")
