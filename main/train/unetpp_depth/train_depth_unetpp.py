import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from USGNet.utils.utils_loss import *
from USGNet.model.unet_pp_TF2.XNet_4 import *
from USGNet.utils.utils_data_generator_onehot_emh import *
from USGNet.utils.utils_metrics import *



#设置为GPU方式
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)


"""
数据处理
"""
####参数
sceneID_train=np.arange(100)
sceneID_test_seen=np.arange(100,130)
sceneID_test_similar=np.arange(130,160)
sceneID_test_novel=np.arange(160,190)
sceneID_all=np.arange(190)
annID=np.arange(256)

camera='kinect'
img_size=224
batch_size=8
epoch=20

save_h5_path='h5/grasp_depth_unetpp.h5'
num_classes=5


####加载路径
##路径
path_x,path_y=load_path(sceneID_train,annID,'grasp_depth')    #只训练训练数据集，训练集又随机划分出测试集

##乱序
np.random.seed(2022)
np.random.shuffle(path_x)
np.random.seed(2022)
np.random.shuffle(path_y)

##分配
n=len(path_x)   #48640
n_train=int(0.8*n)   #38912
train_path_x,train_path_y=path_x[:n_train],path_y[:n_train]
test_path_x,test_path_y=path_x[n_train:],path_y[n_train:]


####数据生成器
train_data_loader=DataSequence_onehot_emh(train_path_x,train_path_y,batch_size,img_size,num_classes,shuffle=True,emh=True)
test_data_loader=DataSequence_onehot_emh(test_path_x,test_path_y,batch_size,img_size,num_classes,shuffle=False,emh=False)



"""
模型处理
"""
####创建模型
model = Xnet(backbone_name='efficientnetb1',
             input_shape=(img_size,img_size, 3),
             encoder_weights='imagenet',
             classes=num_classes,
             activation='softmax')


####编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  #将学习率由0.001调整到0.0005
    loss=dice_focal_loss(10),
    metrics=['accuracy',f_score(),IoU()]
)


####断点续训
checkpoint_save_path='./checkpoint/grasp_depth.ckpt'
if os.path.exists(checkpoint_save_path+'.index'):
    model.load_weights(checkpoint_save_path)

cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                               save_weights_only=True,
                                               monitor='val_loss',
                                               save_best_only=True)


####调节学习率
reduce_lr=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=2,mode='auto')


####训练模型
history=model.fit(
    train_data_loader,
    epochs=epoch,
    validation_data=test_data_loader,
    callbacks=[cp_callback,reduce_lr]
)


####保存模型
# model.save(save_h5_path)
model.save_weights(save_h5_path)  #自定义模型保存出错，使用保存权重的方式保存

####打印模型
model.summary()


####可视化模型
plt.subplot(2,2,1)
plt.plot(history.epoch,history.history['loss'],label='train_loss')
plt.plot(history.epoch,history.history['val_loss'],label='test_loss')
plt.title('The loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2,2,2)
plt.plot(history.epoch,history.history['accuracy'],label='train_acc')
plt.plot(history.epoch,history.history['val_accuracy'],label='test_acc')
plt.title('The accuracy curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(2,2,3)
plt.plot(history.epoch,history.history['_F_Score'],label='train_f_score')
plt.plot(history.epoch,history.history['val__F_Score'],label='test_f_score')
plt.title('The f_score curve')
plt.xlabel('Epoch')
plt.ylabel('F_score')
plt.legend()

plt.subplot(2,2,4)
plt.plot(history.epoch,history.history['_IoU'],label='train_IoU')
plt.plot(history.epoch,history.history['val__IoU'],label='test_IoU')
plt.title('The IoU curve')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()

plt.show()


#####保存训练过程的中间参数
np.save('train_result/unet_train_loss.npy',history.history['loss'])
np.save('train_result/unet_test_loss.npy',history.history['val_loss'])
np.save('train_result/unet_train_acc.npy',history.history['accuracy'])
np.save('train_result/unet_test_acc.npy',history.history['val_accuracy'])

np.save('train_result/unet_train_f_score.npy',history.history['_F_Score'])
np.save('train_result/unet_test_f_score.npy',history.history['val__F_Score'])
np.save('train_result/unet_train_IoU.npy',history.history['_IoU'])
np.save('train_result/unet_test_IoU.npy',history.history['val__IoU'])