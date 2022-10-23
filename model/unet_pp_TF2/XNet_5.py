from mygraspnet15_6dof_newdata.model.unet_pp_TF2.XNet_4 import *
from tensorflow.keras.utils import plot_model

model = Xnet(backbone_name='efficientnetb1',
             input_shape=(None, None, 3),
             encoder_weights='imagenet',
             classes=10, 
             activation=None)
model.summary()

plot_model(model, 'unet++.png', show_shapes=True)