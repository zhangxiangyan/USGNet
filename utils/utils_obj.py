import scipy.io as scio
import numpy as np
import os


"""
加载物体的姿态和物体的id
"""
# def load_obj(sceneId,annId,camera_pose,camera='kinect'):
def load_obj(sceneId,annId,camera='kinect'):
    #加载路径
    root='E:/graspnet/DataSet'
    path=os.path.join(root,'scenes','scene_%04d'%(sceneId),camera,'meta','%04d.mat'%(annId))

    #读入数据
    data=scio.loadmat(path)
    poses=data['poses']  #(3, 4, 9)

    #数据处理
    obj_poses=np.zeros((poses.shape[2],4,4))
    for i in range(poses.shape[2]):
        obj_poses[i,:3,:4]=poses[:,:,i]
        obj_poses[i,3,:]=np.array([0,0,0,1])


    #加载物体id
    path_id=os.path.join(root,'scenes','scene_%04d'%(sceneId),'object_id_list.txt')
    obj_ids=open(path_id).readlines()
    obj_id=[int(i.strip()) for i in obj_ids]


    return obj_id, obj_poses




