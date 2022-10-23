## USGNet: A 6-DoF Grasp Pose Detection Network Based on UNet++ Semantic Segmentation Model
---

## Principle
![alt text](Fig.%204.png)

## Environment
tensorflow==2.6.0

### Dataset
Download the dataset from https://graspnet.net/datasets.html

###Data Conversion
Run USGNet/utils/utils_grasp_to_img.py to to get the semantic image of the grasping parameters.

### Training
Run the UNet++ models with different grasping parameters in sequence, they are stored in the main/train folder.

### 3D visualization
Run USGNet/main/pred/val_all_unetpp_show_pointcloud.py to display predicted grasps in a 3D point cloud.

