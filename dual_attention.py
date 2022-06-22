import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Activation,Input,Conv2D,Softmax,Add,Permute,Dot
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import plot_model


def CAM_Module(x):
    """
       inputs :
            x : input feature maps [B,H,W,C]
       returns :
            out : attention value + input feature
              attention: B X (HxW) X (HxW)
    """

    gamma = tf.Variable(tf.ones(1))

    B,H,W,C= x.shape

    assert K.image_data_format() == "channels_last"

    query = tf.keras.layers.Reshape((-1, C))(x)
    key = tf.keras.layers.Reshape((-1, C))(x)

    energy = tf.linalg.matmul(query, key, transpose_a=True)

    attention = Activation("softmax")(energy)

    value = tf.keras.layers.Reshape((-1, C))(x)
    out = tf.linalg.matmul(attention, value, transpose_b=True)

    out = tf.keras.layers.Reshape(x.shape[1:])(out)

    out = layers.add([gamma * out , x])

    return out

def PAM_Module(x):
    """
       inputs :
            x : input feature maps( B X C X H X W)
       returns :
            out : attention value + input feature
            attention: B X (HxW) X (HxW)
    """

    gamma = tf.Variable(tf.ones(1))

    B, H, W, C = x.shape

    query = Conv2D(filters=C//8,kernel_size=1,padding='same',activation='relu')(x)
    key = Conv2D(filters=C//8,kernel_size=1,padding='same',activation='relu')(x)
    value = Conv2D(filters=C,kernel_size=1,padding='same',activation='relu')(x)

    assert K.image_data_format() == "channels_last"

    query = tf.keras.layers.Reshape((H * W, -1))(query)
    key = tf.keras.layers.Reshape((H * W, -1))(key)

    energy = tf.linalg.matmul(query, key, transpose_b=True)

    attention = Activation("softmax")(energy)

    value = tf.keras.layers.Reshape((H * W, -1))(value)
    out = tf.linalg.matmul(attention,value)

    out = tf.keras.layers.Reshape(x.shape[1:])(out)
    out = layers.Add()([gamma * out , x])

    return out


def danet(inputs):
    x1 = CAM_Module(inputs)
    x2 = PAM_Module(inputs)
    x = layers.add([x1, x2])
    return x

if __name__ == '__main__':
    inputs = keras.Input(shape=[26, 26, 512])
    outputs = danet(inputs)
    model = Model(inputs, outputs)
    model.summary()
    plot_model(model, show_shapes=True, to_file='model_unet.png')


