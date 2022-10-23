################加载库
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model




def conv2(x,filters,ff=True):
    x=Conv2D(filters=filters,kernel_size=3,padding='same',activation='relu')(x)
    if ff:
        x=Conv2D(filters=filters,kernel_size=3,padding='same',activation='relu')(x)
    else:
        x=Conv2D(filters=filters//2,kernel_size=3,padding='same',activation='relu')(x)
    return x


#unet
def unet(x,n_label):
    #左边
    out=[]
    for i in range(4):
        x=conv2(x,2**(i+6))
        out.append(x)
        x=MaxPooling2D(pool_size=2,padding='same')(x)
    #底部
    x=conv2(x,1024)
    #右边
    for i in range(3,-1,-1):
        x=Conv2DTranspose(filters=2**(i+6),kernel_size=2,padding='same',strides=2)(x)
        x=Concatenate()([x,out[i]])
        if i!=0:
            x = conv2(x, 2 ** (i + 6), ff=False)
        else:
            x = conv2(x, 2 ** (i + 6))
    y=Conv2D(filters=n_label,kernel_size=1,padding='same',activation='softmax')(x)
    return y


# #model
# shape=224
# inputs=Input(shape=(shape,shape,3))
# outputs=unet(inputs,2)
# model=tf.keras.Model(inputs=inputs,outputs=outputs)
# model.summary()
# plot_model(model, 'unet_point.png', show_shapes=True)