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

def FCN_32(num_classes,input_height, input_width):
    '''
      [B,H,W,C] = x.shape
    '''
    assert input_height % 32 == 0,'The input_height should be divisible by 32'
    assert input_width  % 32 == 0,'The input_width should be divisible by 32'

    # Returns:A `tensor`.
    inputs=Input(shape=(input_height,input_width,3))

    model = vgg16.VGG16(
        include_top=False,
        weights='imagenet', input_tensor=inputs)
    assert isinstance(model, Model), 'The model should be the instantiation of the keras Model '

    x = Conv2D(
        filters=4096,
        kernel_size=(7, 7),
        padding="same",
        activation="relu",
        name="fc6")(model.output)

    x = Dropout(0.5)(x)

    x = Conv2D(
        filters=4096,
        kernel_size=(1, 1),
        padding="same",
        activation="relu",
        name="fc7")(x)
    x = Dropout(0.5)(x)

    x = Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        padding='same',
        activation='relu')(x)

    # h/32,w/32,21 -> h/16,w/16,21
    x = Conv2DTranspose(filters=num_classes,
                        kernel_size=(2, 2),
                        strides=(2, 2),
                        padding="valid")(x)

    x_ = model.get_layer(name='block4_pool').output
    x_ = Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        padding="same",
        activation="relu")(x_)

    # h/16,w/16,21
    upsample1 = Add()([x_,x])
    upsample1 = Conv2DTranspose(filters=num_classes,
                        kernel_size=(2, 2),
                        strides=(2, 2),
                        padding="valid")(upsample1)


    y_ = model.get_layer(name='block3_pool').output
    y_ = Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        padding="same",
        activation="relu")(y_)

    output = Add()([y_,upsample1])

    upsample2 = Conv2DTranspose(filters=num_classes,
                         kernel_size=(8,8),
                         strides=(8, 8),
                         padding="valid")(output)

    y = Reshape((-1, num_classes))(upsample2)

    y = Activation("softmax")(y)

    fcn_model = Model(inputs=inputs, outputs=y)
    return fcn_model

if __name__ == '__main__':
    model = FCN_32(21, 224, 224)
    model.summary()
    plot_model(model, show_shapes=True, to_file='model_fcn8.png')
    print(len(model.layers))