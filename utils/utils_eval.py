import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K



####计算混淆矩阵
def fast_hist(a,b,n):
    """
    计算混淆矩阵
    :param a: flatten(lab)
    :param b: flatten(pred)
    :param n: num_classes
    :return: hist
    """
    k=(a>=0)&(a<n)    #mask
    #np.bincount 计算每个像素点出现某个种类的次数
    return np.bincount(n*a[k].astype(int)+b[k],minlength=n**2).reshape(n,n)

####计算iou
def per_iou(hist):
    #混淆矩阵中：对角-预测正确的；列-标签值；行-预测值
    return np.diag(hist)/(hist.sum(1)+hist.sum(0)-np.diag(hist))

####计算recall
def per_recall(hist):
    return np.diag(hist)/hist.sum(1)


####计算precision
def per_precision(hist):
    return np.diag(hist)/hist.sum(0)


####计算acc
def per_accuracy(hist):
    return np.sum(np.diag(hist))/np.sum(hist)


####计算f_score
def per_f_score(hist,beta=1,smooth=1e-5):
    return ((1+beta**2)*np.diag(hist)+smooth)/((1+beta**2)*np.diag(hist)+
                                               beta**2*(hist.sum(1)-np.diag(hist))+(hist.sum(0)-np.diag(hist))+smooth)



def compute_m_index(labs,preds,num_classes):
    ##创建空的混淆矩阵
    hist=np.zeros((num_classes,num_classes))

    # ##计算参数
    # for i in range(len(labs)):
    #     #累加每张图的混淆矩阵
    #     hist+=fast_hist(labs[i].flatten(),preds[i].flatten(),num_classes)

    hist = fast_hist(labs, preds, num_classes)   #这只有一张图片，不需要上面的for循环

    prec=per_precision(hist)


    print('mPresision: {}'.format(round(np.nanmean(prec)*100,2)))

    return round(np.nanmean(prec)*100,2)



