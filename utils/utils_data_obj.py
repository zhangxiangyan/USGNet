import numpy as np
import tensorflow as tf
from PIL import Image,ImageEnhance
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
            path_y.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +'/label/{}.png'.format(str(j).zfill(4)))
    return path_x,path_y



##加载深度图路径
def load_depth_path(sceneID,annID,camera='kinect'):
    path_depth=[]
    for i in tqdm(sceneID,desc='Load path:'):
        for j in annID:
            path_depth.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4))+camera+'/depth/{}.png'.format(str(j).zfill(4)))
    return path_depth



##加载单个路径函数
def load_path_all_single(sceneID,annID,camera='kinect'):
    path_rgb='E:/graspnet/DataSet/scenes/scene_{}/'.format(str(sceneID).zfill(4)) + camera + '/rgb/{}.png'.format(str(annID).zfill(4))
    path_seg = 'E:/graspnet/DataSet/scenes/scene_{}/'.format(str(sceneID).zfill(4)) + camera + '/label/{}.png'.format(str(annID).zfill(4))
    path_depth = 'E:/graspnet/DataSet/scenes/scene_{}/'.format(str(sceneID).zfill(4)) + camera + '/depth/{}.png'.format(str(annID).zfill(4))
    return path_rgb,path_seg,path_depth

#
#
# ##生成器类
# class DataSequence_obj(keras.utils.Sequence):
#     def __init__(self,x_path,y_path,batch_size,img_size,cls_num,shuffle=True):
#         self.x_path=x_path
#         self.y_path=y_path
#         self.batch_size=batch_size
#         self.img_size=img_size
#         self.shuffle=shuffle
#         self.cls_num=cls_num
#
#     def __len__(self):
#         return math.ceil(len(self.x_path)/self.batch_size)
#
#     def __getitem__(self,idx):
#         batch_x=self.load_x(self.x_path[idx*self.batch_size:(idx+1)*self.batch_size])  #取idx*batch_size到（idx+1）*batch_size，即取了一个batch的切片
#         batch_y=self.load_y(self.y_path[idx*self.batch_size:(idx+1)*self.batch_size])
#         return np.array(batch_x,dtype=np.float32),np.array(batch_y,dtype=np.int16)
#
#     def on_epoch_end(self):  #每一个epoch结束后将路径数组打乱一次
#         if self.shuffle:
#             np.random.seed(1)
#             np.random.shuffle(self.x_path)
#             np.random.seed(1)
#             np.random.shuffle(self.y_path)
#
#     def load_x(self,path):
#         img_x=[]
#         for i in path:
#             img=Image.open(i)
#             img=img.resize((self.img_size,self.img_size),Image.NEAREST) #邻近采样
#             img=np.array(img)/255.
#             img_x.append(img)
#         return img_x
#
#     def load_y(self,path):
#         img_y=[]
#         for i in path:
#             img=Image.open(i)
#             img=img.resize((self.img_size,self.img_size),Image.NEAREST)
#             img=np.array(img)   #保持物品的分类
#             #转变为onehot
#             img= np.eye(self.cls_num)[img.reshape([-1])]
#             img = img.reshape([self.img_size, self.img_size, self.cls_num])
#             img_y.append(img)
#         return img_y


##生成器类
class DataSequence_obj(keras.utils.Sequence):
    def __init__(self,x_path,y_path,batch_size,img_size,cls_num,shuffle=True,emh=True,eval=False):
        self.x_path=x_path
        self.y_path=y_path
        self.batch_size=batch_size
        self.img_size=img_size
        self.cls_num=cls_num
        self.shuffle=shuffle
        self.emh=emh
        self.eval=eval

    def __len__(self):
        return math.ceil(len(self.x_path)/self.batch_size)

    def __getitem__(self,idx):
        batch_x,batch_y=self.load_x_y(self.x_path[idx*self.batch_size:(idx+1)*self.batch_size],self.y_path[idx*self.batch_size:(idx+1)*self.batch_size])
        return np.array(batch_x, dtype=np.float32), np.array(batch_y)

    def on_epoch_end(self):  #每一个epoch结束后将路径数组打乱一次
        if self.shuffle:
            np.random.seed(1)
            np.random.shuffle(self.x_path)
            np.random.seed(1)
            np.random.shuffle(self.y_path)

    def load_x_y(self,path_x,path_y):
        img_x=[]
        img_y=[]
        for i in range(len(path_x)):
            #打开
            img_in=Image.open(path_x[i])
            img_out=Image.open(path_y[i])

            #缩小
            img_in=img_in.resize((self.img_size,self.img_size),Image.NEAREST)
            img_out = img_out.resize((self.img_size, self.img_size), Image.NEAREST)

            #增强
            if self.emh:
                img_in,img_out=self.data_enhancement(img_in, img_out)

            #变换
            img_in = np.array(img_in) / 255.
            img_out = np.array(img_out)   #这里保持原值

            #独热码
            if self.eval:
                pass
            else:
                img_out=np.eye(self.cls_num)[img_out.reshape([-1])]
                img_out=img_out.reshape([self.img_size,self.img_size,self.cls_num])
            #附加
            img_x.append(img_in)
            img_y.append(img_out)
        return img_x,img_y


    def data_enhancement(self,img_in,img_out):
        #先确定要不要增强
        rand_emh=np.random.rand()
        #增强(p=0.4)
        if rand_emh>0.6:
            rand_time = np.random.randint(2)  # 增强几次

            for i in range(rand_time):
                rand_mode1 = np.random.randint(3)  # 哪种方式增强1
                rand_mode2 = np.random.randint(3)  # 哪种方式增强2
                rand_factor = np.random.randint(8, 12) / 10  # 因子
                # 方式1
                if rand_mode1 == 0:
                    img_in = ImageEnhance.Brightness(img_in).enhance(factor=rand_factor)
                elif rand_mode1 == 1:
                    img_in = ImageEnhance.Color(img_in).enhance(factor=rand_factor)
                else:
                    pass  #空操作
                #方式2
                if rand_mode2 == 0:
                    img_in = img_in.transpose(Image.FLIP_LEFT_RIGHT)
                    img_out = img_out.transpose(Image.FLIP_LEFT_RIGHT)
                elif rand_mode2 == 1:
                    img_in = img_in.transpose(Image.FLIP_TOP_BOTTOM)
                    img_out = img_out.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    pass  #空操作

        #不增强（p=0.6)
        else:
            pass

        return img_in, img_out



