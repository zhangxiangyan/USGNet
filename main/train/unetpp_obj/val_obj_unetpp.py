import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from USGNet.utils.utils_loss import *
from USGNet.utils.utils_data_obj import *
from USGNet.model.unet_pp_TF2.XNet_4 import *
from PIL import Image



#设置为GPU方式
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


"""
数据处理
"""
####参数
sceneID_train=np.arange(100)
sceneID_test_seen=np.arange(100,130)
sceneID_test_similar=np.arange(130,160)
sceneID_test_novel=np.arange(160,190)
sceneID_all=np.arange(190)
annID=np.arange(256)

camera='kinect'
img_size=224

save_h5_path='h5/grasp_obj_unetpp.h5'
num_classes=89


####加载路径
##路径
path_x,path_y=load_path(sceneID_all,annID)

##乱序
np.random.seed(2022)
np.random.shuffle(path_x)
np.random.seed(2022)
np.random.shuffle(path_y)


##分配(6:2:2)
n=len(path_x)   #48640
n_train=int(0.6*n)
n_test=int(0.8*n)


# test_path_x,test_path_y=path_x[n_test:n_test+2],path_y[n_test:n_test+2]   #只加载两张
test_path_x,test_path_y=path_x[n_test+2:n_test+4],path_y[n_test+2:n_test+4]   #只加载两张

##图像处理函数
def load_x(path):
    img_x = []
    for i in path:
        img = Image.open(i)
        img = img.resize((img_size,img_size), Image.NEAREST)  # 邻近采样
        img = np.array(img) / 255.
        img_x.append(img)
    return np.array(img_x,dtype=np.float32)

def load_y(path):
    img_y = []
    for i in path:
        img = Image.open(i)
        img = img.resize((img_size, img_size), Image.NEAREST)
        img = np.array(img)
        img_y.append(img)
    return np.array(img_y,dtype=np.int16)


##得到数据
test_x=load_x(test_path_x)
test_y=load_y(test_path_y)


"""
模型处理
"""
####加载模型
model = Xnet(backbone_name='efficientnetb1',
             input_shape=(img_size,img_size, 3),
             encoder_weights='imagenet',
             classes=num_classes,
             activation='softmax')
model.load_weights(save_h5_path)


pred=model.predict(test_x)
pred=np.argmax(pred,axis=-1)



"""
可视化
"""
num=2
for i in range(num):
    plt.subplot(num,3,1+3*i)
    plt.imshow(test_x[i])
    if i==0:
        plt.title('Image')

    plt.subplot(num,3,2+3*i)
    plt.imshow(test_y[i],cmap=plt.get_cmap('CMRmap'))
    if i == 0:
        plt.title('Label')

    plt.subplot(num,3,3+3*i)
    # plt.imshow(pred[i],cmap=plt.get_cmap('CMRmap'))
    plt.imshow(pred[i], cmap='magma')
    if i == 0:
        plt.title('Prediction')

plt.show()