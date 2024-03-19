import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import SimpleITK as sitk
import os
import torch
import matplotlib.pyplot as plt
#from typing import List, Union, Tuple

import torchio as tio
           ###########  Dataloader  #############
NUM_WORKERS=0
PIN_MEMORY=True
DIM_ = 256
   
  
    
def crop_center_3D(img,cropx=DIM_,cropy=DIM_):
    z,x,y = img.shape
    startx = x//2 - cropx//2
    starty = (y)//2 - cropy//2    
    return img[:,startx:startx+cropx, starty:starty+cropy]

def Cropping_3d(org_dim3,org_dim1,org_dim2,DIM_,img_):# org_dim3->numof channels
    
    if org_dim1<DIM_ and org_dim2<DIM_:
        padding1=int((DIM_-org_dim1)//2)
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,padding1:org_dim1+padding1,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = temp
    if org_dim1>DIM_ and org_dim2>DIM_:
        img_ = crop_center_3D(img_)        
        ## two dims are different ####
    if org_dim1<DIM_ and org_dim2>=DIM_:
        padding1=int((DIM_-org_dim1)//2)
        temp=np.zeros([org_dim3,DIM_,org_dim2])
        temp[:,padding1:org_dim1+padding1,:] = img_[:,:,:]
        img_=temp
        img_ = crop_center_3D(img_)
    if org_dim1==DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,DIM_,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_=temp
    
    if org_dim1>DIM_ and org_dim2<DIM_:
        padding2=int((DIM_-org_dim2)//2)
        temp=np.zeros([org_dim3,org_dim1,DIM_])
        temp[:,:,padding2:org_dim2+padding2] = img_[:,:,:]
        img_ = crop_center_3D(temp)   
    return img_


def Normalization_1(img):
        mean=np.mean(img)
        std=np.std(img)
        img=(img-mean)/std
        return img 

geometrical_transforms = tio.OneOf([
    tio.RandomFlip(axes=([1, 2])),
    #tio.RandomElasticDeformation(num_control_points=(5, 5, 5), locked_borders=1, image_interpolation='nearest'),
    tio.RandomAffine(degrees=(-45, 45), center='image'),
])

intensity_transforms = tio.OneOf([
    tio.RandomBlur(),
    tio.RandomGamma(log_gamma=(-0.2, -0.2)),
    tio.RandomNoise(mean=0.1, std=0.1),
    tio.RandomGhosting(axes=([1, 2])),
])

transforms_2d = tio.Compose({
    geometrical_transforms: 0.3,  # Probability for geometric transforms
    intensity_transforms: 0.3,   # Probability for intensity transforms
    tio.Lambda(lambda x: x): 0.4 # Probability for no augmentation (original image)
})
   
def generate_label(gt):
        temp_ = np.zeros([4,DIM_,DIM_])
        temp_[0:1,:,:][np.where(gt==1)]=1
        temp_[1:2,:,:][np.where(gt==2)]=1
        temp_[2:3,:,:][np.where(gt==3)]=1
        temp_[3:4,:,:][np.where(gt==0)]=1
        return temp_



class Dataset_val(Dataset): 
    def __init__(self, images_folder):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3)) 
        print(img_path)
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]

        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-11]+'_gt.nii.gz'
        print(gt_path)
        
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        gt = gt.astype(np.float64)
        
                
        return img,gt
        
def Data_Loader_val(images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

val_imgs = r'C:\My_Data\TMI\Second_Data\Only_4Chamber\imgs/' ## path to images
train_loader = Data_Loader_val(val_imgs,batch_size = 1)
a = iter(train_loader)

for i in range(1):
    a1 =next(a) 

# for i in range(82):
#     a1 =next(a) 

#     img = a1[0].numpy()
#     gt = a1[1].numpy()

#     plt.figure()
#     plt.imshow(gt[0,:])
    
#     plt.figure()
#     plt.imshow(img[0,:])
    

# img = sitk.ReadImage(r"C:\My_Data\TMI\Second_Data\New_Data\F1\val\P004\cine_lax_forlabel.nii.gz")    ## --> [H,W,C]
# img1 = sitk.GetArrayFromImage(img)  

# for i in range(1):
#     plt.figure()
#     plt.imshow(img1[i,:])
    
# for i in range(3):
#     plt.figure()
#     plt.imshow(img1[i,:])

# img = sitk.ReadImage(r"C:\My_Data\TMI\Second_Data\New_Data\F1\val\P019\cine_lax_label.nii.gz")    ## --> [H,W,C]
# img2 = sitk.GetArrayFromImage(img)   

# for i in range(3):
#     plt.figure()
#     plt.imshow(img2[i,:])

