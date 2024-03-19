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
   
  
def resample_itk_image_LA(itk_image):
    # Get original spacing and size
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()
    
    out_spacing = (1,1,1)

    # Calculate new size
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # Instantiate resample filter with properties
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkNearestNeighbor)

    # Execute resampling
    resampled_image = resample.Execute(itk_image)
    return resampled_image


    
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


class Dataset_io(Dataset): 
    def __init__(self, images_folder,transformations=transforms_2d):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        self.transformations = transformations
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3))
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = resample_itk_image_LA(img)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]
        img = Normalization_1(img)
        C = img.shape[0]
        H = img.shape[1]
        W = img.shape[2]
        img = Cropping_3d(C,H,W,256,img)
        img = np.expand_dims(img, axis=3)
        
        gt_path = os.path.join(self.gt_folder,str(self.images_name[index]).zfill(3))
        gt_path = gt_path[:-7]+'_gt.nii.gz'
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt)      ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        C = gt.shape[0]
        H = gt.shape[1]
        W = gt.shape[2]
        gt = Cropping_3d(C,H,W,256,gt)
        gt = np.expand_dims(gt, axis=3)
        
        d = {}
        d['Image'] = tio.Image(tensor = img, type=tio.INTENSITY)
        d['Mask'] = tio.Image(tensor = gt, type=tio.LABEL)
        sample = tio.Subject(d)
        if self.transformations is not None:
            transformed_tensor = self.transformations(sample)
            img = transformed_tensor['Image'].data
            gt = transformed_tensor['Mask'].data
    
        gt = gt[...,0]
        img = img[...,0] 
        gt = generate_label(gt)
        return img,gt
        
def Data_Loader_io_transforms(images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_io(images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader


class Dataset_val(Dataset): 
    def __init__(self, images_folder):  ## If I apply Data Augmentation here, the validation loss becomes None. 
        self.images_folder = images_folder
        self.gt_folder = self.images_folder[:-5] + 'gts'
        
        self.images_name = os.listdir(images_folder)
        
    def __len__(self):
       return len(self.images_name)
    def __getitem__(self, index):
        img_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3)) + '/cine_lax_forlabel.nii.gz'
        img = sitk.ReadImage(img_path)    ## --> [H,W,C]
        img = resample_itk_image_LA(img)      ## --> [H,W,C]
        img = sitk.GetArrayFromImage(img)   ## --> [C,H,W]

        gt_path = os.path.join(self.images_folder,str(self.images_name[index]).zfill(3)) + '/cine_lax_label.nii.gz'
        gt = sitk.ReadImage(gt_path)    ## --> [H,W,C]
        gt = resample_itk_image_LA(gt)      ## --> [H,W,C]
        gt = sitk.GetArrayFromImage(gt)   ## --> [C,H,W]
        gt = gt.astype(np.float64)
        
        name = self.images_name[index]
        print(name)
        
        for i in range(gt.shape[0]):
            gt_ps = gt[i,:]
            unique_values = np.unique(gt_ps)
            num_unique_values = len(unique_values)
            if num_unique_values==5:
                new_img = img[i,:]
                new_gt = gt[i,:]
                
                new_img = sitk.GetImageFromArray(new_img)
                new_gt = sitk.GetImageFromArray(new_gt)
                
                sitk.WriteImage(new_img,r'C:\My_Data\TMI\Second_Data\Only_4Chamber\imgs'+'/'+name+'_img'+'.nii.gz')
                sitk.WriteImage(new_gt,r'C:\My_Data\TMI\Second_Data\Only_4Chamber\gts'+'/'+name+'_gt'+'.nii.gz')
                
        return  0
        
def Data_Loader_val(images_folder,batch_size,num_workers=NUM_WORKERS,pin_memory=PIN_MEMORY):
    test_ids = Dataset_val(images_folder=images_folder)
    data_loader = DataLoader(test_ids,batch_size=batch_size,num_workers=num_workers,pin_memory=pin_memory,shuffle=True)
    return data_loader

val_imgs = r'C:\My_Data\TMI\Second_Data\New_Data/' ## path to images
train_loader = Data_Loader_val(val_imgs,batch_size = 1)
a = iter(train_loader)

for i in range(103):
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

