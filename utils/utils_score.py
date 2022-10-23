import numpy as np
import tensorflow as tf


def score_all(c_p,c_r,c_a,c_d,c_w,p_p,p_r,p_a,p_d,p_w,n_p=2,n_r=256,n_a=256,n_d=5,n_w=17,w1=0.5,w2=0.5):
    """
    计算综合分数，用于选出综合分数考的抓取
    :param c_: 预测的分类
    :param p_: 预测的分类对应的概率值
    :param n_: 分类数
    :return: 综合考虑5个抓取参数的分数
    """
    ##拷贝值，以免改变原有的值
    b_p=c_p.copy()   #用复制的方式可以避免复制的值修改后对原值的影响
    b_r=c_r.copy()
    b_a=c_a.copy()
    b_d=c_d.copy()
    b_w=c_w.copy()

    ##把预测分类转换为二值分类（背景，非背景）
    b_p[b_p>1]=1    #这里这样写是错的，因为这会改变原值
    b_r[b_r>1]=1
    b_a[b_a>1]=1
    b_d[b_d>1]=1
    b_w[b_w>1]=1

    ##归一化系数
    max=n_p+n_r+n_a+w1*n_d+w2*n_w
    min=0

    ##分数
    score=((b_p*b_r*b_a*b_d*b_w)*(n_p*p_p+n_r*p_r+n_a*p_a+w1*n_d*p_d+w2*n_w*p_w)-min)/(max-min)

    return score



####去除物体以外的抓取
def score_all_objbi(c_obj,c_p,c_r,c_a,c_d,c_w,p_p,p_r,p_a,p_d,p_w,n_p=2,n_r=256,n_a=256,n_d=5,n_w=17,w1=0.5,w2=0.5):
    """
    计算综合分数，用于选出综合分数考的抓取
    :param c_: 预测的分类
    :param p_: 预测的分类对应的概率值
    :param n_: 分类数
    :return: 综合考虑5个抓取参数的分数
    """
    ##拷贝值，以免改变原有的值
    b_obj=c_obj.copy()
    b_p=c_p.copy()   #用复制的方式可以避免复制的值修改后对原值的影响
    b_r=c_r.copy()
    b_a=c_a.copy()
    b_d=c_d.copy()
    b_w=c_w.copy()

    ##把预测分类转换为二值分类（背景，非背景）
    b_obj[b_obj>1]=1
    b_p[b_p>1]=1
    b_r[b_r>1]=1
    b_a[b_a>1]=1
    b_d[b_d>1]=1
    b_w[b_w>1]=1

    ##归一化系数
    max=n_p+n_r+n_a+w1*n_d+w2*n_w
    min=0

    ##分数
    score=((b_obj*b_p*b_r*b_a*b_d*b_w)*(n_p*p_p+n_r*p_r+n_a*p_a+w1*n_d*p_d+w2*n_w*p_w)-min)/(max-min)

    return score





####特定分类的分数
def score_all_obj(obj,obj_id,c_p,c_r,c_a,c_d,c_w,p_p,p_r,p_a,p_d,p_w,n_p=2,n_r=256,n_a=256,n_d=5,n_w=17,w1=0.5,w2=0.5):
    ##拷贝值，以免改变原有的值
    b_p=c_p.copy()   #用复制的方式可以避免复制的值修改后对原值的影响
    b_r=c_r.copy()
    b_a=c_a.copy()
    b_d=c_d.copy()
    b_w=c_w.copy()

    ##把预测分类转换为二值分类（背景，非背景）
    b_p[b_p>1]=1
    b_r[b_r>1]=1
    b_a[b_a>1]=1
    b_d[b_d>1]=1
    b_w[b_w>1]=1

    ##筛选出特定对象的抓取参数
    mask=(obj!=obj_id)
    b_p[mask]=0
    b_r[mask]=0
    b_a[mask]=0
    b_d[mask]=0
    b_w[mask]=0

    ##归一化系数
    max=n_p+n_r+n_a+w1*n_d+w2*n_w
    min=0

    ##分数
    score=((b_p*b_r*b_a*b_d*b_w)*(n_p*p_p+n_r*p_r+n_a*p_a+w1*n_d*p_d+w2*n_w*p_w)-min)/(max-min)

    return score



####特定分类的分数
def score_all_mul_obj(obj,c_p,c_r,c_a,c_d,c_w,p_p,p_r,p_a,p_d,p_w,n_p=2,n_r=256,n_a=256,n_d=5,n_w=17,w1=0.5,w2=0.5,obj_id=[9,10,19,20,21]):
    ##拷贝值，以免改变原有的值
    b_p=c_p.copy()   #用复制的方式可以避免复制的值修改后对原值的影响
    b_r=c_r.copy()
    b_a=c_a.copy()
    b_d=c_d.copy()
    b_w=c_w.copy()

    ##把预测分类转换为二值分类（背景，非背景）
    b_p[b_p>1]=1
    b_r[b_r>1]=1
    b_a[b_a>1]=1
    b_d[b_d>1]=1
    b_w[b_w>1]=1

    ##筛选出特定对象的抓取参数
    mask=((obj!=obj_id[0])&(obj!=obj_id[1])&(obj!=obj_id[2])&(obj!=obj_id[3])&(obj!=obj_id[4]))
    b_p[mask]=0
    b_r[mask]=0
    b_a[mask]=0
    b_d[mask]=0
    b_w[mask]=0

    ##归一化系数
    max=n_p+n_r+n_a+w1*n_d+w2*n_w
    min=0

    ##分数
    score=((b_p*b_r*b_a*b_d*b_w)*(n_p*p_p+n_r*p_r+n_a*p_a+w1*n_d*p_d+w2*n_w*p_w)-min)/(max-min)

    return score