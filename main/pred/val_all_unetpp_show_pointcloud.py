import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from USGNet.utils.utils_loss import *
from USGNet.model.unet_pp_TF2.XNet_4 import *
from USGNet.utils.utils_data_generator_onehot_emh import *
from USGNet.utils.utils_metrics import *
from USGNet.utils.utils_show import *
from USGNet.utils.utils_grasp import *
from USGNet.utils.utils_obj import *
from USGNet.utils.utils_view import *
from USGNet.utils.utils_score import *
from scipy.spatial.transform import Rotation as R


#设置为GPU方式
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


"""
数据处理
"""
####参数
# sceneID, annID=110,0
# sceneID, annID=140,0
# sceneID, annID=170,0
sceneID, annID=150,0

score_threshold=0.1

camera='kinect'
img_size=224

####加载路径
path_rgb=load_path_all_single_2(sceneID, annID)


##图像处理函数
def load_rgb(path):
    img = Image.open(path)
    img = img.resize((img_size, img_size), Image.NEAREST)  # 邻近采样
    img = np.array(img) / 255
    return np.array(img,dtype=np.float32).reshape((1,img_size,img_size,3))

##得到数据
test_rgb=load_rgb(path_rgb)


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

##预测
pred_point=model1.predict(test_rgb)
pred_view=model2.predict(test_rgb)
pred_ang=model3.predict(test_rgb)
pred_depth=model4.predict(test_rgb)
pred_width=model5.predict(test_rgb)
pred_obj=model6.predict(test_rgb)

##预测分类的概率值
pred_point_p=np.max(pred_point,axis=-1).reshape((img_size,img_size))  #分类的最大概率
pred_view_p=np.max(pred_view,axis=-1).reshape((img_size,img_size))
pred_ang_p=np.max(pred_ang,axis=-1).reshape((img_size,img_size))
pred_depth_p=np.max(pred_depth,axis=-1).reshape((img_size,img_size))
pred_width_p=np.max(pred_width,axis=-1).reshape((img_size,img_size))
pred_obj_p=np.max(pred_obj,axis=-1).reshape((img_size,img_size))

##预测分类
pred_point=np.argmax(pred_point,axis=-1).reshape((img_size,img_size))
pred_view=np.argmax(pred_view,axis=-1).reshape((img_size,img_size))
pred_ang=np.argmax(pred_ang,axis=-1).reshape((img_size,img_size))
pred_depth=np.argmax(pred_depth,axis=-1).reshape((img_size,img_size))
pred_width=np.argmax(pred_width,axis=-1).reshape((img_size,img_size))
pred_obj=np.argmax(pred_obj,axis=-1).reshape((img_size,img_size))


"""
图像放大
"""
def arr2img_enlarge(arr):
    arr=arr.astype(np.uint8)   #必须要加这一行，否则会报错
    img=Image.fromarray(arr)
    img=img.resize((1280,720),Image.NEAREST)
    return np.array(img).reshape(-1)

pred_point=arr2img_enlarge(pred_point)
pred_view=arr2img_enlarge(pred_view)
pred_ang=arr2img_enlarge(pred_ang)
pred_depth=arr2img_enlarge(pred_depth)
pred_width=arr2img_enlarge(pred_width)
pred_obj=arr2img_enlarge(pred_obj)

pred_point_p=(arr2img_enlarge(pred_point_p*255))/255
pred_view_p=(arr2img_enlarge(pred_view_p*255))/255
pred_ang_p=(arr2img_enlarge(pred_ang_p*255))/255
pred_depth_p=(arr2img_enlarge(pred_depth_p*255))/255
pred_width_p=(arr2img_enlarge(pred_width_p*255))/255
pred_obj_p=(arr2img_enlarge(pred_obj_p*255))/255


"""
抓取综合分数
"""
eval_score=score_all_objbi(pred_obj,pred_point,pred_view,pred_ang,pred_depth,pred_width,
                pred_point_p,pred_view_p,pred_ang_p,pred_depth_p,pred_width_p,
                n_p=2,n_r=65,n_a=37,n_d=5,n_w=17,w1=0.5,w2=0.5)

print(np.unique(eval_score))

"""
抓取参数转换为真实值
"""
cloud = create_pcd_from_depth(sceneID, annID, camera=camera).reshape([-1, 3])
mask=((eval_score!=0)&(eval_score>score_threshold)&(cloud[:,2]!=0.))   #筛除背景+设置分数阈值+筛除离群点
grasp_points=cloud[mask]
grasp_depth=pred_depth[mask]
grasp_width=pred_width[mask]
grasp_vec_view=pred_view[mask]
grasp_vec_ang=pred_ang[mask]

grasp_points[:,2]-=0.005 #因为抓取点采样时，z值超过了物体表面，所以加一个补偿值

scores=eval_score[mask]

##其他抓取参数
#字典
depth_txt=np.array([0,0.01,0.02,0.03,0.04],dtype=np.float32)  #dtype=np.float32一定要加类型否则会报错
width_txt=np.array([0.,0.,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15],dtype=np.float32)
depth_dict={num:txt for num,txt in enumerate(depth_txt)}  #通过这个变换，将回归值转为分类值
width_dict={num:txt for num,txt in enumerate(width_txt)}
view=np.load(r'E:\Data\python_program\USGNet\utils\view_64.npy')
ang=(np.arange(36) * (np.pi / 36))

#分类转数值
grasp_depth=np.array([depth_dict[i] for i in grasp_depth],dtype=np.float32)
grasp_width=np.array([width_dict[i] for i in grasp_width],dtype=np.float32)
grasp_vec_view=np.array([view[i-1] for i in grasp_vec_view],dtype=np.float32)
grasp_vec_ang=np.array([ang[i-1] for i in grasp_vec_ang],dtype=np.float32)
grasp_vec=grasp_vec_view*(grasp_vec_ang.reshape(-1,1))

##生成旋转矩阵
r=R.from_rotvec(grasp_vec)
Rs_true=r.as_matrix()


"""
可视化
"""
####可视化
gripper=[]

# 生成夹爪
print('预测的抓取个数：',len(grasp_points))
for i in range(len(grasp_points)):
    gripper.append(mesh_gripper(grasp_points[i],Rs_true[i],grasp_width[i],grasp_depth[i]-0.01, scores[i]))


#显示
shuffled_gripper=random_sample(gripper, numGrasp=50)
pcd = creat_scene_cloudpoint(sceneID, annID,camera=camera)
o3d.visualization.draw_geometries([pcd, *shuffled_gripper])
