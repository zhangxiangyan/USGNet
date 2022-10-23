import open3d as o3d
from utils_obj import *
from utils_grasp import *
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

#相关参数
root='E:/graspnet/DataSet'
camera='kinect'
# camera='realsense'


num_view=64
num_ang=36


"""
分类对应字典
"""
#分类字典
in_plane_rot_txt=np.array([0.,0.2617994,0.5235988,0.7853982,1.0471976,1.3089969,1.5707964,1.8325957,2.0943952,2.3561945,2.6179938,2.8797932],dtype=np.float32)
depth_txt=np.array([0.01,0.02,0.03,0.04],dtype=np.float32)  #dtype=np.float32一定要加类型否则会报错
width_txt=np.array([0.,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15],dtype=np.float32)

in_plane_dict={txt:num for num,txt in enumerate(in_plane_rot_txt)}  #通过这个变换，将回归值转为分类值
depth_dict={txt:num for num,txt in enumerate(depth_txt)}  #通过这个变换，将回归值转为分类值
width_dict={txt:num for num,txt in enumerate(width_txt)}  #通过这个变换，将回归值转为分类值。(np.rint((np.unique(width))*100))/100匹配前需要先变换



"""
将Rs_vec存为两张图，vec+ang
"""
def Rs_vec_to_cls(Rs_vec):
    ####生成view的图
    view = np.load('view_{}.npy'.format(num_view))  # 上半球的分类

    # 计算余弦相似度（两向量夹角的余弦值）
    cos_dis = cosine_similarity(Rs_vec, view)
    cos_argmax = np.argmax(cos_dis, axis=-1)
    cos_argmax = np.uint8(cos_argmax)

    ####生成ang的图
    # 计算旋转向量的模长
    norm = np.linalg.norm(Rs_vec, ord=2, axis=-1).reshape(-1, 1)  # 计算模长

    # 模长
    a = (np.arange(num_ang) * (np.pi / num_ang)).reshape(-1, 1)

    # 计算距离
    dis = euclidean_distances(norm, a)
    dis_argmin = np.argmin(dis, axis=-1)
    dis_argmin = np.uint8(dis_argmin)

    return cos_argmax,dis_argmin



"""
点的坐标和颜色数组生成点云
"""
def arr_to_pcd(points,color):
    #生成点云
    pcd = o3d.geometry.PointCloud()
    #坐标点
    pcd.points = o3d.utility.Vector3dVector(points)
    #颜色
    pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd



"""
加载相机内参
"""
def get_camera_parameters(camera='kinect'):
    import open3d as o3d
    param = o3d.camera.PinholeCameraParameters()
    param.extrinsic = np.eye(4, dtype=np.float64)
    if camera == 'kinect':
        param.intrinsic.set_intrinsics(1280, 720, 631.5, 631.2, 639.5, 359.5)
    elif camera == 'realsense':
        param.intrinsic.set_intrinsics(1280, 720, 927.17, 927.37, 639.5, 359.5)
    return param



"""
点云转rgb图
"""
def grasp_point_to_rgb_img(pcd,name,sceneId,annId,camera='kinect',show=False,root='E:/graspnet/DataSet'):
    path= os.path.join(root, 'scenes', 'scene_%04d' % sceneId, camera, 'grasp_label_new')
    save_path=os.path.join(path, name, '{}_{}.png'.format(name,str(annId).zfill(4)))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720)
    vis.get_render_option().background_color = np.array([0, 0, 0])  # 把背景设置为黑色
    vis.get_render_option().point_size = 15  # 设置渲染点的大小
    vis.add_geometry(pcd)  # 加载点云
    vis.get_view_control().convert_from_pinhole_camera_parameters(get_camera_parameters(camera=camera))  # 加载相机参数
    if show:
        vis.run()  # 显示点云
    else:
        vis.capture_screen_image(save_path, do_render=True)   #存储为图片



"""
得到转化后的抓取参数，并转存为图像
"""
def load_grasp_save(sceneId,annId,camera='kinect',fric_coef_thresh=0.1):
    #####加载抓取参数
    num_views, num_angles, num_depths = 300, 12, 4
    #加载物体参数
    obj_list, pose_list = load_obj(sceneId, annId,camera=camera)

    #加载碰撞标签
    collision_dump=loadCollisionLabel(sceneId)

    #生成接近向量
    template_views = np.load('views.npy')   #(1, 300, 12, 4, 3)


    #加载抓取标签
    grasp_points=[]
    grasp_vec=[]
    grasp_depth=[]
    grasp_width=[]
    for i,(obj_idx,trans) in enumerate(zip(obj_list, pose_list)):
        #加载数据
        sampled_points, offsets, fric_coefs = loadGraspLabel(obj_idx)
        collision = collision_dump[i]  # 某场景下第i个物体的碰撞情况，用于确定非碰撞的抓取点

        #数据分配
        target_points = sampled_points[:, np.newaxis, np.newaxis, np.newaxis, :]  # 扩充点的维度（n,3）--(n,1,1,1,3)
        target_points = np.tile(target_points, [1, num_views, num_angles, num_depths, 1])  # (n,1,1,1,3)--(n,300,12,4,3)

        views = np.tile(template_views, [sampled_points.shape[0], 1, 1, 1, 1])

        angles = offsets[:, :, :, :, 0]  # offset分别存储平面内旋转角度，深度，宽度
        depths = offsets[:, :, :, :, 1]
        widths = offsets[:, :, :, :, 2]

        #筛选数据（mask不仅查找所需项，还能够降维）
        mask=((fric_coefs<=fric_coef_thresh)&(fric_coefs>0)&(~collision))
        target_points = target_points[mask]

        if (len(target_points))==0:
            continue
        else:
            views = views[mask]
            angles = angles[mask]
            depths = depths[mask]
            widths = widths[mask]
            #####转换抓取参数
            target_points = transform_points(target_points, trans)

            Rs = batch_viewpoint_params_to_matrix(-views, angles)  # 通过view采样点和平面内的抓取确定出旋转矩阵
            Rs = np.matmul(trans[np.newaxis, :3, :3], Rs)  # 物品的姿态*抓取的姿态

            widths=np.around(widths,2).astype(np.float32)

            #Rs转换为旋转向量
            r = R.from_matrix(Rs)
            Rs_vec = r.as_rotvec()


        #抓取参数+字典：回归-分类
        for j in range(len(target_points)):
            grasp_points.append(target_points[j, :])
            grasp_vec.append(Rs_vec[j,:])
            grasp_depth.append(depth_dict[depths[j]])
            grasp_width.append(width_dict[widths[j]])

    grasp_points=np.array(grasp_points)
    grasp_vec=np.array(grasp_vec)
    grasp_depth=np.array(grasp_depth)
    grasp_width=np.array(grasp_width)

    grasp_r,grasp_theta=Rs_vec_to_cls(grasp_vec)  #将旋转向量分解为方向和角度的分类



    #####存储抓取参数为图像
    #点云颜色
    arr_ones = np.ones(len(grasp_points))
    grasp_points_color = np.stack([arr_ones, arr_ones, arr_ones], axis=-1)

    grasp_r_change = (1 / num_view) * (grasp_r + 1)
    grasp_r_change_color = np.stack([grasp_r_change, grasp_r_change, grasp_r_change], axis=-1)

    grasp_theta_change = (1 / num_ang) * (grasp_theta + 1)
    grasp_theta_change_color = np.stack([grasp_theta_change, grasp_theta_change,grasp_theta_change], axis=-1)

    grasp_depth_change = (1 / 4) * (grasp_depth + 1)
    grasp_depth_color = np.stack([grasp_depth_change, grasp_depth_change, grasp_depth_change], axis=-1)

    grasp_width_change = (1 / 16) * (grasp_width + 1)
    grasp_width_color = np.stack([grasp_width_change, grasp_width_change, grasp_width_change], axis=-1)


    #点云
    grasp_points_pcd = arr_to_pcd(grasp_points, grasp_points_color)
    grasp_r_pcd = arr_to_pcd(grasp_points, grasp_r_change_color)
    grasp_theta_pcd = arr_to_pcd(grasp_points,grasp_theta_change_color)
    grasp_depth_pcd = arr_to_pcd(grasp_points, grasp_depth_color)
    grasp_width_pcd = arr_to_pcd(grasp_points, grasp_width_color)


    #存图
    grasp_point_to_rgb_img(grasp_points_pcd, 'grasp_points', sceneId, annId, camera=camera)
    grasp_point_to_rgb_img(grasp_r_pcd, 'grasp_vec_view', sceneId, annId, camera=camera)
    grasp_point_to_rgb_img(grasp_theta_pcd, 'grasp_vec_ang', sceneId, annId, camera=camera)
    grasp_point_to_rgb_img(grasp_depth_pcd, 'grasp_depth', sceneId, annId, camera=camera)
    grasp_point_to_rgb_img(grasp_width_pcd, 'grasp_width', sceneId, annId, camera=camera)




"""
调用函数，转存为图像
"""
# load_grasp_save(0,0,camera=camera,fric_coef_thresh=0.1)

sceneID=np.arange(190)
annID=np.arange(256)

for i in tqdm(sceneID,desc='save point to image'):
    for j in annID:
        load_grasp_save(i,j, camera=camera, fric_coef_thresh=0.1)






