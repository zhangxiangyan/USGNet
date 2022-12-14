import numpy as np
import tensorflow as tf
from PIL import Image
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import tensorflow.keras as keras  #加上这个才能自动补全keras的函数



##加载路径函数
def load_path(sceneID,annID,camera='kinect'):
    path_x=[]
    path_y=[]
    for i in tqdm(sceneID,desc='Load path:'):
        for j in annID:
            path_x.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4))+camera+'/rgb/{}.png'.format(str(j).zfill(4)))
            path_y.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_all/png_file_segm/grasp_points/grasp_points_rgb_{}.png'.format(str(j).zfill(4)))
    return path_x,path_y



##生成器类
class DataSequence(keras.utils.Sequence):
    def __init__(self,x_path,y_path,batch_size,img_size,shuffle=True):
        self.x_path=x_path
        self.y_path=y_path
        self.batch_size=batch_size
        self.img_size=img_size
        self.shuffle=shuffle

    def __len__(self):
        return math.ceil(len(self.x_path)/self.batch_size)

    def __getitem__(self,idx):
        batch_x=self.load_x(self.x_path[idx*self.batch_size:(idx+1)*self.batch_size])  #取idx*batch_size到（idx+1）*batch_size，即取了一个batch的切片
        batch_y=self.load_y(self.y_path[idx*self.batch_size:(idx+1)*self.batch_size])
        return np.array(batch_x,dtype=np.float32),np.array(batch_y,dtype=np.int8)

    def on_epoch_end(self):  #每一个epoch结束后将路径数组打乱一次
        if self.shuffle:
            np.random.seed(1)
            np.random.shuffle(self.x_path)
            np.random.seed(1)
            np.random.shuffle(self.y_path)

    def load_x(self,path):
        img_x=[]
        for i in path:
            img=Image.open(i)
            img=img.resize((self.img_size,self.img_size),Image.NEAREST) #邻近采样
            img=np.array(img)/255.
            img_x.append(img)
        return img_x

    def load_y(self,path):
        img_y=[]
        for i in path:
            img=Image.open(i)
            img=img.resize((self.img_size,self.img_size),Image.NEAREST)
            img=np.array(img)[:,:,0]/255
            img_y.append(img)
        return img_y