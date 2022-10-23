
from __future__ import absolute_import

from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_unet_2d import unet_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_vnet_2d import vnet_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_unet_plus_2d import unet_plus_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_r2_unet_2d import r2_unet_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_att_unet_2d import att_unet_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_resunet_a_2d import resunet_a_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_u2net_2d import u2net_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_unet_3plus_2d import unet_3plus_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_transunet_2d import transunet_2d
from mygraspnet15_6dof_newdata.model.keras_unet_collection._model_swin_unet_2d import swin_unet_2d



# model=transunet_2d((224,224,3),[64, 128, 256, 512],2)
# model=u2net_2d((224,224,3),2,[64, 128, 256, 512],output_activation='Softmax')
# model=swin_unet_2d((224,224,3), 64, n_labels, depth, stack_num_down, stack_num_up,
#                       patch_size, num_heads, window_size, num_mlp
model.summary()