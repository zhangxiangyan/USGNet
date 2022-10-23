import numpy as np
import tensorflow as tf
from PIL import Image,ImageEnhance
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import tensorflow.keras as keras  #加上这个才能自动补全keras的函数



####加载路径函数
def load_path_all_eval(sceneID,annID,camera='kinect'):
    path_rgb=[]
    path_point=[]
    path_view = []
    path_ang = []
    path_depth = []
    path_width = []
    path_obj=[]

    file_name1='grasp_points'
    file_name2 = 'grasp_vec_view'
    file_name3= 'grasp_vec_ang'
    file_name4 = 'grasp_depth'
    file_name5 = 'grasp_width'

    for i in tqdm(sceneID,desc='Load path:'):
        for j in annID:
            path_rgb.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4))+camera+'/rgb/{}.png'.format(str(j).zfill(4)))
            path_point.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name1,file_name1,str(j).zfill(4)))
            path_view.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name2,file_name2,str(j).zfill(4)))
            path_ang.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name3,file_name3,str(j).zfill(4)))
            path_depth.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name4,file_name4,str(j).zfill(4)))
            path_width.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name5,file_name5,str(j).zfill(4)))
            path_obj.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4))+camera+'/label/{}.png'.format(str(j).zfill(4)))

    return path_rgb,path_point,path_view,path_ang,path_depth,path_width,path_obj


####加载路径函数
def load_path_all_eval_64_36(sceneID,annID,camera='kinect'):
    path_rgb=[]
    path_point=[]
    path_view = []
    path_ang = []
    path_depth = []
    path_width = []
    path_obj=[]

    file_name1='grasp_points'
    file_name2 = 'grasp_vec_view_64'
    file_name3= 'grasp_vec_ang_36'
    file_name4 = 'grasp_depth'
    file_name5 = 'grasp_width'

    for i in tqdm(sceneID,desc='Load path:'):
        for j in annID:
            path_rgb.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4))+camera+'/rgb/{}.png'.format(str(j).zfill(4)))
            path_point.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name1,file_name1,str(j).zfill(4)))
            path_view.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name2,file_name2,str(j).zfill(4)))
            path_ang.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name3,file_name3,str(j).zfill(4)))
            path_depth.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name4,file_name4,str(j).zfill(4)))
            path_width.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4)) + camera +
                          '/grasp_label_new/{}/{}_{}.png'.format(file_name5,file_name5,str(j).zfill(4)))
            path_obj.append('E:/graspnet/DataSet/scenes/scene_{}/'.format(str(i).zfill(4))+camera+'/label/{}.png'.format(str(j).zfill(4)))

    return path_rgb,path_point,path_view,path_ang,path_depth,path_width,path_obj


####迭代器（多输出迭代器）
##生成器类
class DataSequence_onehot_emh_all(keras.utils.Sequence):
    def __init__(self,x_path,point_path,view_path,ang_path,depth_path,width_path,obj_path,
                 batch_size,img_size,cls_num_p,cls_num_v,cls_num_a,cls_num_d,cls_num_w,cls_num_obj,
                 shuffle=True,emh=True,eval=False):
        self.x_path=x_path
        self.point_path=point_path
        self.view_path = view_path
        self.ang_path = ang_path
        self.depth_path = depth_path
        self.width_path = width_path
        self.obj_path=obj_path

        self.batch_size=batch_size
        self.img_size=img_size

        self.cls_num_p=cls_num_p
        self.cls_num_v = cls_num_v
        self.cls_num_a = cls_num_a
        self.cls_num_d = cls_num_d
        self.cls_num_w = cls_num_w
        self.cls_num_obj = cls_num_obj

        self.shuffle=shuffle
        self.emh=emh
        self.eval=eval

    def __len__(self):
        return math.ceil(len(self.x_path)/self.batch_size)


    def __getitem__(self,idx):
        # batch_x,batch_y=self.load_x_y(self.x_path[idx*self.batch_size:(idx+1)*self.batch_size],self.point_path[idx*self.batch_size:(idx+1)*self.batch_size])
        batch_x,batch_p,batch_v,batch_a,batch_d,batch_w,batch_obj=self.load_x_y(self.x_path[idx*self.batch_size:(idx+1)*self.batch_size],
                       self.point_path[idx*self.batch_size:(idx+1)*self.batch_size],self.view_path[idx*self.batch_size:(idx+1)*self.batch_size],
                       self.ang_path[idx*self.batch_size:(idx+1)*self.batch_size],self.depth_path[idx*self.batch_size:(idx+1)*self.batch_size],
                       self.width_path[idx*self.batch_size:(idx+1)*self.batch_size],self.obj_path[idx*self.batch_size:(idx+1)*self.batch_size])

        return np.array(batch_x, dtype=np.float32), {'point':np.array(batch_p),'view':np.array(batch_v),'ang':np.array(batch_v),
                                                     'depth':np.array(batch_d),'width':np.array(batch_w),'object':np.array(batch_obj)}



    def on_epoch_end(self):  #每一个epoch结束后将路径数组打乱一次
        if self.shuffle:
            np.random.seed(1)
            np.random.shuffle(self.x_path)
            np.random.seed(1)
            np.random.shuffle(self.point_path)
            np.random.seed(1)
            np.random.shuffle(self.view_path)
            np.random.seed(1)
            np.random.shuffle(self.ang_path)
            np.random.seed(1)
            np.random.shuffle(self.depth_path)
            np.random.seed(1)
            np.random.shuffle(self.width_path)
            np.random.seed(1)
            np.random.shuffle(self.obj_path)

    def load_x_y(self,path_x,path_point,path_view,path_ang,path_depth,path_width,path_obj):
        img_x=[]
        img_y_p=[]
        img_y_v = []
        img_y_a = []
        img_y_d = []
        img_y_w = []
        img_y_obj = []

        for i in range(len(path_x)):
            #打开
            img_in=Image.open(path_x[i])
            img_point=Image.open(path_point[i])
            img_view = Image.open(path_view[i])
            img_ang = Image.open(path_ang[i])
            img_depth = Image.open(path_depth[i])
            img_width = Image.open(path_width[i])
            img_obj = Image.open(path_obj[i])

            #缩小
            img_in=img_in.resize((self.img_size,self.img_size),Image.NEAREST)
            img_point = img_point.resize((self.img_size, self.img_size), Image.NEAREST)
            img_view = img_view.resize((self.img_size, self.img_size), Image.NEAREST)
            img_ang = img_ang.resize((self.img_size, self.img_size), Image.NEAREST)
            img_depth = img_depth.resize((self.img_size, self.img_size), Image.NEAREST)
            img_width = img_width.resize((self.img_size, self.img_size), Image.NEAREST)
            img_obj = img_obj.resize((self.img_size, self.img_size), Image.NEAREST)

            #增强
            if self.emh:
                img_in,img_point,img_view,img_ang,img_depth,img_width,img_obj=self.data_enhancement(img_in,img_point,img_view,img_ang,img_depth,img_width,img_obj)

            #变换
            img_in = np.array(img_in) / 255.
            img_point = np.int16(np.rint((np.array(img_point)[:, :, 0] / 255)*(self.cls_num_p-1)))
            img_view = np.int16(np.rint((np.array(img_view)[:, :, 0] / 255) * (self.cls_num_v - 1)))
            img_ang = np.int16(np.rint((np.array(img_ang)[:, :, 0] / 255) * (self.cls_num_a - 1)))
            img_depth = np.int16(np.rint((np.array(img_depth)[:, :, 0] / 255) * (self.cls_num_d - 1)))
            img_width = np.int16(np.rint((np.array(img_width)[:, :, 0] / 255) * (self.cls_num_w - 1)))
            # img_obj = np.int16(np.rint((np.array(img_obj)[:, :, 0] / 255) * (self.cls_num_obj - 1)))
            img_obj = np.int16((np.array(img_obj)))

            #独热码
            if self.eval:
                pass
            else:
                img_point=np.eye(self.cls_num_p)[img_point.reshape([-1])]
                img_point=img_point.reshape([self.img_size,self.img_size,self.cls_num_p])

                img_view=np.eye(self.cls_num_v)[img_view.reshape([-1])]
                img_view=img_view.reshape([self.img_size,self.img_size,self.cls_num_v])

                img_ang=np.eye(self.cls_num_a)[img_ang.reshape([-1])]
                img_ang=img_ang.reshape([self.img_size,self.img_size,self.cls_num_a])

                img_depth=np.eye(self.cls_num_d)[img_depth.reshape([-1])]
                img_depth=img_depth.reshape([self.img_size,self.img_size,self.cls_num_d])

                img_width=np.eye(self.cls_num_w)[img_width.reshape([-1])]
                img_width=img_width.reshape([self.img_size,self.img_size,self.cls_num_w])

                img_obj=np.eye(self.cls_num_obj)[img_obj.reshape([-1])]
                img_obj=img_obj.reshape([self.img_size,self.img_size,self.cls_num_obj])


            #附加
            img_x.append(img_in)
            img_y_p.append(img_point)
            img_y_v.append(img_view)
            img_y_a.append(img_ang)
            img_y_d.append(img_depth)
            img_y_w.append(img_width)
            img_y_obj.append(img_obj)

        return img_x,img_y_p,img_y_v,img_y_a,img_y_d,img_y_w,img_y_obj


    def data_enhancement(self,img_in,img_point,img_view,img_ang,img_depth,img_width,img_obj):
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
                    img_point = img_point.transpose(Image.FLIP_LEFT_RIGHT)
                    img_view = img_view.transpose(Image.FLIP_LEFT_RIGHT)
                    img_ang = img_ang.transpose(Image.FLIP_LEFT_RIGHT)
                    img_depth = img_depth.transpose(Image.FLIP_LEFT_RIGHT)
                    img_width = img_width.transpose(Image.FLIP_LEFT_RIGHT)
                    img_obj = img_obj.transpose(Image.FLIP_LEFT_RIGHT)

                elif rand_mode2 == 1:
                    img_in = img_in.transpose(Image.FLIP_TOP_BOTTOM)
                    img_point = img_point.transpose(Image.FLIP_TOP_BOTTOM)
                    img_view = img_view.transpose(Image.FLIP_TOP_BOTTOM)
                    img_ang = img_ang.transpose(Image.FLIP_TOP_BOTTOM)
                    img_depth = img_depth.transpose(Image.FLIP_TOP_BOTTOM)
                    img_width = img_width.transpose(Image.FLIP_TOP_BOTTOM)
                    img_obj = img_obj.transpose(Image.FLIP_TOP_BOTTOM)
                else:
                    pass  #空操作

        #不增强（p=0.6)
        else:
            pass

        return img_in,img_point,img_view,img_ang,img_depth,img_width,img_obj








