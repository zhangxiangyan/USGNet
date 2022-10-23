import numpy as np


def generate_views(N,center=np.zeros(3,dtype=np.float32),r=1,phi=((np.sqrt(5)-1))/2):
    """
    斐波那契网格采样
    :param N: 采样点数
    :param center: 采样中心
    :param r: 球体搬家
    :param phi: 黄金分割比例，约为0.618=(5^0.5-1)/2
    :return:采样点坐标和接近方向
    """
    n=np.arange(N,dtype=np.float32)
    new_reg=np.array([-1,1])  #把z放缩到（-1，1）之间
    z=((n-np.min(n))/(np.max(n)-np.min(n)))*(np.max(new_reg)-np.min(new_reg))+np.min(new_reg)
    x=np.sqrt(1-z**2)*np.cos(2*np.pi*n*phi)
    y=np.sqrt(1-z**2)*np.sin(2*np.pi*n*phi)
    views=np.stack([x,y,z],axis=1)  #按列进行堆叠
    views=r*np.array(views)+center
    return views   #单位求上的点的坐标，平方相加等于1，即可作为方向向量


# view=generate_views(300)
# np.save('view_300.npy',view)

# view=np.load('view_300.npy')
# print(view.shape)
# print(view[:5])