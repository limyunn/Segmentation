import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Model,layers,Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation
from tensorflow.keras.applications import vgg16,resnet50
import numpy as np
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
import math
from keras_flops import get_flops
import tensorflow_addons as tfa
from dual_attention import *



def Unet(input_shape=(224,224,3), num_classes=21,freeze=None):
    '''
      [B,H,W,C] = x.shape
    '''
    #(None, 224, 224, 3)
    img_input=Input(shape=input_shape)

    assert img_input.shape[1] % 32 == 0,\
        'The input_height should be divisible by 32'
    assert img_input.shape[2] % 32 == 0,\
        'The input_width should be divisible by 32'

    # -------------------------------------------------#
    #              Encoder
    # -------------------------------------------------#
    vgg_model = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=img_input)


    assert isinstance(vgg_model, Model), 'The model should be the instantiation of the keras Model '

    # -------------------------------------------------#
    #              Decoder
    # -------------------------------------------------#
    # 14,14,512 -> 28,28,512 -> 28,28,1024 -> 28,28,512
    P5 = UpSampling2D((2, 2))(vgg_model.get_layer(name='block5_conv3').output)
    att = danet(vgg_model.get_layer(name="block4_conv3").output)
    P5 = concatenate([att, P5], axis=-1)
    P5 = Conv2D(512, (3, 3), activation='relu', padding="same", kernel_initializer = RandomNormal(stddev=0.02))(P5)
    P5 = Conv2D(512, (3, 3), activation='relu', padding="same", kernel_initializer = RandomNormal(stddev=0.02))(P5)


    # 28,28,512 -> 56,56,512 -> 56,56,768 -> 56,56,256
    P4 = UpSampling2D((2, 2))(P5)
    P4 = concatenate([vgg_model.get_layer(
        name="block3_conv3").output, P4], axis=-1)
    P4 = Conv2D(256, (3, 3), activation='relu', padding="same", kernel_initializer = RandomNormal(stddev=0.02))(P4)
    P4 = Conv2D(256, (3, 3), activation='relu', padding="same", kernel_initializer = RandomNormal(stddev=0.02))(P4)


    # 56,56,256 -> 112,112,256 -> 112,112,384 -> 112,112,128
    P3 = UpSampling2D((2, 2))(P4)
    P3 = concatenate([vgg_model.get_layer(
        name="block2_conv2").output, P3], axis=-1)
    P3 = Conv2D(128, (3, 3), activation='relu', padding="same", kernel_initializer = RandomNormal(stddev=0.02))(P3)
    P3 = Conv2D(128, (3, 3), activation='relu', padding="same", kernel_initializer = RandomNormal(stddev=0.02))(P3)


    # 112,112,128 -> 224,224,128 -> 224,224,192 -> 224,224,64
    P2 = UpSampling2D((2, 2))(P3)
    P2 = concatenate([vgg_model.get_layer(
        name="block1_conv2").output, P2], axis=-1)
    P2 = Conv2D(64, (3, 3), activation='relu', padding="same", kernel_initializer = RandomNormal(stddev=0.02))(P2)
    P2 = Conv2D(64, (3, 3), activation='relu', padding="same", kernel_initializer = RandomNormal(stddev=0.02))(P2)
    P1 = Conv2D(num_classes, (1, 1), padding="same")(P2)
    P1 = Activation('softmax')(P1)

    return Model(inputs=img_input, outputs=P1)



if __name__ == '__main__':
    model= Unet()
    from tensorflow.keras.utils import plot_model
    plot_model(model, show_shapes=True, to_file='model_unet.png')

    print(model.layers[18].gamma)#取出pam_module中可训练的gamma值

    #<tf.Variable 'pam__module/pam_gamma:0' shape=(1,) dtype=float32, numpy=array([-0.06657953], dtype=float32)>
    for _,layer in enumerate(model.layers):
        print(_,layer.name)
    model.summary()
    # flops = get_flops(Unet(), batch_size=1) / (2 * 10 ** 9)
    # print(f"FLOPs:{flops:.1f}G")
