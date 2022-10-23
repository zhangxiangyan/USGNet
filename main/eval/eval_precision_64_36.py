import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from USGNet.utils.utils_loss import *
from USGNet.model.unet_pp_TF2.XNet_4 import *
from USGNet.utils.utils_data_generator_onehot_emh import *
from USGNet.utils.utils_data_eval import *
from USGNet.utils.utils_metrics import *
from USGNet.utils.utils_show import *
from USGNet.utils.utils_grasp import *
from USGNet.utils.utils_obj import *
from USGNet.utils.utils_view import *
from USGNet.utils.utils_score import *
from scipy.spatial.transform import Rotation as R
from USGNet.utils.utils_eval import *
import time

#设置为GPU方式
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

a=time.time()
"""
数据处理
"""
####参数
# file_name='test_seen'
# file_name='test_similar'
file_name='test_novel'

sceneID_train=np.arange(100)
sceneID_test_seen=np.arange(100,130)
sceneID_test_similar=np.arange(130,160)
sceneID_test_novel=np.arange(160,190)
annID=np.arange(256)

camera='kinect'
img_size=224
batch_size=8

###加载数据
# path_rgb,path_point,path_view,path_ang,path_depth,path_width,path_obj=load_path_all_eval_64_36(sceneID_test_seen,annID,camera='kinect')
# path_rgb,path_point,path_view,path_ang,path_depth,path_width,path_obj=load_path_all_eval_64_36(sceneID_test_similar,annID,camera='kinect')
path_rgb,path_point,path_view,path_ang,path_depth,path_width,path_obj=load_path_all_eval_64_36(sceneID_test_novel,annID,camera='kinect')


test_data_loader=DataSequence_onehot_emh_all(path_rgb,path_point,path_view,path_ang,path_depth,path_width,path_obj,
                            batch_size,img_size,2,65,37,5,17,89,shuffle=False,emh=False)



"""
模型处理
"""
####加载模型
model1=Xnet(backbone_name='efficientnetb1',input_shape=(img_size,img_size, 3),classes=2,activation='softmax')
model1.load_weights('h5/grasp_point_unetpp_1.h5')
model2=Xnet(backbone_name='efficientnetb1',input_shape=(img_size,img_size, 3),classes=65,activation='softmax')
model2.load_weights('h5/grasp_view_64_unetpp.h5')
model3=Xnet(backbone_name='efficientnetb1',input_shape=(img_size,img_size, 3),classes=37,activation='softmax')
model3.load_weights('h5/grasp_ang_36_unetpp.h5')
model4=Xnet(backbone_name='efficientnetb1',input_shape=(img_size,img_size, 3),classes=5,activation='softmax')
model4.load_weights('h5/grasp_depth_unetpp.h5')
model5=Xnet(backbone_name='efficientnetb1',input_shape=(img_size,img_size, 3),classes=17,activation='softmax')
model5.load_weights('h5/grasp_width_unetpp.h5')
model6=Xnet(backbone_name='efficientnetb1',input_shape=(img_size,img_size, 3),classes=89,activation='softmax')
model6.load_weights('h5/grasp_obj_unetpp.h5')



def arr2img_enlarge(arr):  ####一张图像放大
    arr=arr.astype(np.uint8)   #必须要加这一行，否则会报错
    img=Image.fromarray(arr)
    img=img.resize((1280,720),Image.NEAREST)
    return np.array(img).reshape(-1)

def arr2img_enlarge_array(arr):  ####一组图像放大
    arr=arr.astype(np.uint8)   #必须要加这一行，否则会报错
    img_arr=np.uint8(np.zeros((arr.shape[0],1280*720)))
    for i in range(len(arr)):
        img=Image.fromarray(arr[i])
        img=img.resize((1280,720),Image.NEAREST)
        img_arr[i,:]=np.array(img).reshape(-1)
    return img_arr


prec=[]

for batch_x,batch_dict in test_data_loader:    #用于测试迭代器
    ##预测
    pred_point = model1.predict(batch_x)
    pred_view = model2.predict(batch_x)
    pred_ang = model3.predict(batch_x)
    pred_depth = model4.predict(batch_x)
    pred_width = model5.predict(batch_x)
    pred_obj = model6.predict(batch_x)


    ##预测分类的概率值
    pred_point_p=np.max(pred_point,axis=-1).reshape((batch_size,img_size,img_size))  #分类的最大概率
    pred_view_p=np.max(pred_view,axis=-1).reshape((batch_size,img_size,img_size))
    pred_ang_p=np.max(pred_ang,axis=-1).reshape((batch_size,img_size,img_size))
    pred_depth_p=np.max(pred_depth,axis=-1).reshape((batch_size,img_size,img_size))
    pred_width_p=np.max(pred_width,axis=-1).reshape((batch_size,img_size,img_size))
    pred_obj_p=np.max(pred_obj,axis=-1).reshape((batch_size,img_size,img_size))

    ##预测分类
    pred_point=np.argmax(pred_point,axis=-1).reshape((batch_size,img_size,img_size))
    pred_view=np.argmax(pred_view,axis=-1).reshape((batch_size,img_size,img_size))
    pred_ang=np.argmax(pred_ang,axis=-1).reshape((batch_size,img_size,img_size))
    pred_depth=np.argmax(pred_depth,axis=-1).reshape((batch_size,img_size,img_size))
    pred_width=np.argmax(pred_width,axis=-1).reshape((batch_size,img_size,img_size))
    pred_obj=np.argmax(pred_obj,axis=-1).reshape((batch_size,img_size,img_size))

    ##实际标签
    true_point_p = np.max(batch_dict['point'], axis=-1).reshape((batch_size, img_size, img_size))
    true_view_p = np.max(batch_dict['view'], axis=-1).reshape((batch_size, img_size, img_size))
    true_ang_p =np.max(batch_dict['ang'], axis=-1).reshape((batch_size, img_size, img_size))
    true_depth_p = np.max(batch_dict['depth'], axis=-1).reshape((batch_size, img_size, img_size))
    true_width_p = np.max(batch_dict['width'], axis=-1).reshape((batch_size, img_size, img_size))
    true_obj_p = np.max(batch_dict['object'], axis=-1).reshape((batch_size, img_size, img_size))

    true_point = np.argmax(batch_dict['point'], axis=-1).reshape((batch_size, img_size, img_size))
    true_view = np.argmax(batch_dict['view'], axis=-1).reshape((batch_size, img_size, img_size))
    true_ang = np.argmax(batch_dict['ang'], axis=-1).reshape((batch_size, img_size, img_size))
    true_depth = np.argmax(batch_dict['depth'], axis=-1).reshape((batch_size, img_size, img_size))
    true_width = np.argmax(batch_dict['width'], axis=-1).reshape((batch_size, img_size, img_size))
    true_obj = np.argmax(batch_dict['object'], axis=-1).reshape((batch_size, img_size, img_size))


    """
    抓取综合分数
    """
    score_pred=score_all_objbi(pred_obj,pred_point,pred_view,pred_ang,pred_depth,pred_width,
                    pred_point_p,pred_view_p,pred_ang_p,pred_depth_p,pred_width_p,
                    n_p=2,n_r=65,n_a=37,n_d=5,n_w=17,w1=0.5,w2=0.5)

    score_true=score_all_objbi(true_obj,true_point,true_view,true_ang,true_depth,true_width,
                    true_point_p,true_view_p,true_ang_p,true_depth_p,true_width_p,
                    n_p=2,n_r=65,n_a=37,n_d=5,n_w=17,w1=0.5,w2=0.5)

    mask=(true_obj>0)   #去除背景的预测
    score_true=np.int8(score_true[mask])
    score_pred[score_pred>0]=1
    score_pred=np.int8(score_pred[mask])



    prec.append(compute_m_index(score_true,score_pred, 2))



prec=np.array(prec)
np.save('eval_result/AP_{}_zdy_64_36.npy'.format(file_name),prec)

print('**********************')
print('AP:',np.mean(-np.sort(-prec)[:50]))
print('**********************')


b=time.time()

print('time:',b-a)
