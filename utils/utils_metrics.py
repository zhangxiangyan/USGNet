import numpy as np
from tensorflow.keras import backend as K



####IoU=tp/(tp+fn+fp)
def IoU(threshold=0.5,smooth=1e-5):
    def _IoU(y_true,y_pred):
        y_pred=K.greater(y_pred,threshold)
        y_pred=K.cast(y_pred,K.floatx())

        tp=K.sum(y_pred*y_true,axis=[0,1,2])
        fn=K.sum(y_true,axis=[0,1,2])-tp
        fp=K.sum(y_pred,axis=[0,1,2])-tp

        iou=(tp+smooth)/(tp+fn+fp+smooth)
        return iou
    return _IoU




####f_score=(1+β^2)*p*r/(β^2*p+r)=(1+β^2)*tp/(1+β^2)*tp+β^2*fn+fp
def f_score(threshold=0.5,beta=0.5,smooth=1e-5):
    def _F_Score(y_true,y_pred):
        y_pred=K.greater(y_pred,threshold) #将被转变为布尔值
        y_pred=K.cast(y_pred,K.floatx()) #把布尔值转换为数值

        tp=K.sum(y_true*y_pred,axis=[0,1,2])
        fn=K.sum(y_true,axis=[0,1,2])-tp
        fp=K.sum(y_pred,axis=[0,1,2])-tp

        score=((1+beta**2)*tp+smooth)/((1+beta**2)*tp+beta**2*fn+fp+smooth)  #加smooth是为了防止分母为0
        return score
    return _F_Score















