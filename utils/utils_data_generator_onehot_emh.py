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
def load_path(sceneID,annID,file_name,camera='kinect'):
    path_x=[]
    path_y=[]
    for i in tqdm(sceneID,desc='Load path:'):
        for j in annID:
            path_x.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4))+camera+'/rgb/{}.png'.format(str(j).zfill(4)))
            path_y.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name,file_name,str(j).zfill(4)))
    return path_x,path_y


##加载单个路径函数
def load_path_all_single(sceneID,annID,camera='kinect'):
    path = 'E:/graspnet/DataSet/scenes/scene_{}/'.format(str(sceneID).zfill(4)) + camera + '/grasp_label_new/'

    path_rgb='E:/graspnet/DataSet/scenes/scene_{}/'.format(str(sceneID).zfill(4)) + camera + '/rgb/{}.png'.format(str(annID).zfill(4))
    path_point=path+'grasp_points/grasp_points_{}.png'.format(str(annID).zfill(4))
    path_view=path + 'grasp_vec_view/grasp_vec_view_{}.png'.format(str(annID).zfill(4))
    path_ang=path + 'grasp_vec_ang/grasp_vec_ang_{}.png'.format(str(annID).zfill(4))
    path_depth=path + 'grasp_depth/grasp_depth_{}.png'.format(str(annID).zfill(4))
    path_width=path + 'grasp_width/grasp_width_{}.png'.format(str(annID).zfill(4))

    return path_rgb,path_point,path_view,path_ang,path_depth,path_width


##加载单个路径函数2
def load_path_all_single_2(sceneID,annID,camera='kinect'):
    path = 'E:/graspnet/DataSet/scenes/scene_{}/'.format(str(sceneID).zfill(4)) + camera + '/grasp_label_new/'

    path_rgb='E:/graspnet/DataSet/scenes/scene_{}/'.format(str(sceneID).zfill(4)) + camera + '/rgb/{}.png'.format(str(annID).zfill(4))
    # path_point=path+'grasp_points/grasp_points_{}.png'.format(str(annID).zfill(4))
    # path_view=path + 'grasp_vec_view_32/grasp_vec_view_32_{}.png'.format(str(annID).zfill(4))
    # path_ang=path + 'grasp_vec_ang_36/grasp_vec_ang_36_{}.png'.format(str(annID).zfill(4))
    # path_depth=path + 'grasp_depth/grasp_depth_{}.png'.format(str(annID).zfill(4))
    # path_width=path + 'grasp_width/grasp_width_{}.png'.format(str(annID).zfill(4))
    #
    # return path_rgb,path_point,path_view,path_ang,path_depth,path_width
    return path_rgb


##生成器类
class DataSequence_onehot_emh(keras.utils.Sequence):
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
            img_out = np.int16(np.rint((np.array(img_out)[:, :, 0] / 255)*(self.cls_num-1)))

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




