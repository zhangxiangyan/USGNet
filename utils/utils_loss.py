import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

"""
必须要把标签转化为one-hot才能用
"""

####cross_entropy
def ce_loss():
    def _CE_Loss(y_true,y_pred):
        y_pred=K.clip(y_pred,K.epsilon(),1.0-K.epsilon())

        CE_loss=-y_true*K.log(y_pred)
        CE_loss=K.mean(K.sum(CE_loss),axis=-1)
        return CE_loss
    return _CE_Loss


####cross_entropy with weight
def ce_weight_loss(cls_weights):
    cls_weights=np.reshape(cls_weights,[1,1,1,-1])
    def _CE_Weight_Loss(y_true,y_pred):
        y_pred=K.clip(y_pred,K.epsilon(),1.0-K.epsilon())

        CE_W_loss=-y_true*K.log(y_pred)*cls_weights
        CE_W_loss=K.mean(K.sum(CE_W_loss),axis=-1)
        return CE_W_loss
    return _CE_Weight_Loss


####focal_loss=-(1-pt)^r*log(pt)
def focal_loss(cls_weights,alpha=0.5,gamma=2):
    cls_weights=np.reshape(cls_weights,[1,1,1,-1])  #这为什么是四维：样本序号、长、宽、分类通道数的0或1的值？
    def _Focal_Loss(y_true,y_pred):  #为了单独传入真值和预测值而新建的内部调用的函数
        #限定范围
        y_pred=K.clip(y_pred,K.epsilon(),1.0-K.epsilon())  #clip用来限制最大最小范围

        #y.log(pred)
        logpt=-y_true*K.log(y_pred)*cls_weights
        logpt=-K.sum(logpt,axis=-1)   #计算所有分类的总和。  #这要乘-1吗，是的，因为整体加起来就一个负号
        # logpt = K.sum(logpt, axis=-1)

        #pt
        pt=tf.exp(logpt)   #这为什么不用K.exp？  用K是对整个批次进行操作；而用tf是对单个张量进行操作

        #alpha
        if alpha is not None:
            logpt*=alpha

        #cross_entropy
        CE_loss=-((1-pt)**gamma)*logpt   #这里有负号会抵消之前的负号
        CE_loss=K.mean(CE_loss)
        return CE_loss
    return _Focal_Loss



####dice_loss=1-2tp/(2tp+fp+fn)
def dice_loss(smooth=1e-5):   #最好不要单独用，否则不收敛
    def _Dice_Loss(y_true,y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # tp=K.sum(y_true*y_pred,axis=[0,1,2])
        # fp=K.sum(y_pred,axis=[0,1,2])-tp
        # fn=K.sum(y_true,axis=[0,1,2])-tp
        tp=K.sum(y_true*y_pred,axis=[0,1,2])
        fp=K.sum(y_pred,axis=[0,1,2])-tp
        fn=K.sum(y_true,axis=[0,1,2])-tp

        score=(2*tp+smooth)/(2*tp+fp+fn+smooth)    #加smooth是为了避免分母出现0而报错
        score=tf.reduce_mean(score)

        dice_loss=1-score
        return dice_loss
    return _Dice_Loss


####dice_ce_w_loss
def dice_ce_w_loss(cls_weights,smooth=1e-5):
    cls_weights=np.reshape(cls_weights,[1,1,1,-1])
    def Dice_CE_Loss(y_true,y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        #ce
        ce_loss=-y_true*K.log(y_pred)*cls_weights
        ce_loss=K.mean(K.sum(ce_loss),axis=-1)

        #dice
        tp=K.sum(y_true*y_pred,axis=[0,1,2])
        fp=K.sum(y_pred,axis=[0,1,2])-tp
        fn=K.sum(y_true,axis=[0,1,2])-tp

        score=(2*tp+smooth)/(2*tp+fp+fn+smooth)
        score=tf.reduce_mean(score)

        dice_loss=1-score
        return ce_loss+dice_loss
    return Dice_CE_Loss


####dice_focal_loss
def dice_focal_loss(cls_weights,alpha=0.5,gamma=2,smooth=1e-5):
    cls_weights=np.reshape(cls_weights,[1,1,1,-1])
    def _Dice_Focal_Loss(y_true,y_pred):
        y_pred=K.clip(y_pred,K.epsilon(),1-K.epsilon())

        #focal_loss
        logpt=-y_true*K.log(y_pred)*cls_weights
        logpt=-K.sum(logpt,axis=-1)  #总共3个负号，就相当于是1个负号

        if alpha is not None:
            logpt*=alpha

        pt=tf.exp(logpt)  #pt=e(log(pt))

        focal_loss=-((1-pt)**gamma)*logpt
        focal_loss=K.mean(focal_loss)

        #dice_loss
        tp=K.sum(y_true*y_pred,axis=[0,1,2])
        fp=K.sum(y_pred,axis=[0,1,2])-tp
        fn=K.sum(y_true,axis=[0,1,2])-tp

        score=(2*tp+smooth)/(2*tp+fp+fn+smooth)
        score=tf.reduce_mean(score)
        dice_loss=1-score

        return focal_loss+dice_loss
    return _Dice_Focal_Loss
