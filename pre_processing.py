# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 14:08:02 2021

@author: Administrator
"""
#loading ML dependancies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nibabel as nib
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import segmentation_models as sm
focal_loss = sm.losses.cce_dice_loss
from nilearn.image import resample_img
import SimpleITK as sitk

##Checking Shape of image
img=nib.load(r'C:\Users\Administrator\Downloads\nfbs\NFBS_Dataset\A00060848\sub-A00060848_ses-NFB3_T1w.nii.gz')
print('Shape of image=',img.shape)


#storing each type of images in a list for easy working
brain_mask=[]
brain=[]
raw=[]
for subdir, dirs, files in os.walk(r'C:\Users\Administrator\Downloads\nfbs\NFBS_Dataset'):
    for file in files:
        #print os.path.join(subdir, file)y
        filepath = subdir + os.sep + file

        if filepath.endswith(".gz"):
          if '_brainmask.' in filepath:
            brain_mask.append(filepath)
          elif '_brain.' in filepath:
            brain.append(filepath)
          else:
            raw.append(filepath)


#creating a dataframe for ease of use
data=pd.DataFrame({'brain_mask':brain_mask,'brain':brain,'raw':raw})
data.head()


##Storing images after performing Bias_Feild_Correction  
bias_correction_path = r'C:\Users\Administrator\Downloads\nfbs\bias_correction'

"""Bias Feild Correction:
Because of magnetic field variations, intensity of images changes it order to retify we use Bias Feild Correction
Intensity Normalisation:
It helps to converge faster by removing scale in-variance
Resizing:
Reducing pixels size for complete fitting images into model
"""

class preprocessing():
  def __init__(self,df):
    self.data=df
    self.raw_index=[]
    self.mask_index=[]
  def bias_correction(self):
 #   !mkdir bias_correction
    for i in tqdm(range(len(self.data))):
        img = sitk.ReadImage(self.data.raw.iloc[i])
        img_mask = sitk.OtsuThreshold(img)
        img = sitk.Cast(img, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        img_c = corrector.Execute(img, img_mask)
        outputImageFileName = bias_correction_path + '/' + str(i)+'.nii.gz'
        sitk.WriteImage(img_c, outputImageFileName)
    index_corr=['bias_correction/'+str(i)+'.nii.gz' for i in range(125)]
    data['bias_corr']=index_corr
    print('Bias corrected images stored at : bias_correction/')
  def resize_crop(self):
    #Reducing the size of image due to memory constraints
   # !mkdir resized
    target_shape = np.array((96,128,160))                   #reducing size of image from 256*256*192 to 96*128*160
    new_resolution = [2,]*3
    new_affine = np.zeros((4,4))
    new_affine[:3,:3] = np.diag(new_resolution)
    # putting point 0,0,0 in the middle of the new volume - this could be refined in the future
    new_affine[:3,3] = target_shape*new_resolution/2.*-1
    new_affine[3,3] = 1.
    raw_index=[]
    mask_index=[]
    #resizing both image and mask and storing in folder
    for i in range(len(data)):
      downsampled_and_cropped_nii = resample_img(self.data.bias_corr.iloc[i], target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
      downsampled_and_cropped_nii.to_filename('resized/raw'+str(i)+'.nii.gz')
      self.raw_index.append('resized/raw'+str(i)+'.nii.gz')
      downsampled_and_cropped_nii = resample_img(self.data.brain_mask.iloc[i], target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
      downsampled_and_cropped_nii.to_filename('resized/mask'+str(i)+'.nii.gz')
      self.mask_index.append('resized/mask'+str(i)+'.nii.gz')
    return self.raw_index,self.mask_index
  def intensity_normalization():
    for i in raw_index:
      image = sitk.ReadImage(i)
      resacleFilter = sitk.RescaleIntensityImageFilter()
      resacleFilter.SetOutputMaximum(255)
      resacleFilter.SetOutputMinimum(0)
      image = resacleFilter.Execute(image)
      sitk.WriteImage(image,i)
    print('Normalization done. Images stored at: resized/')


pre=preprocessing(data)
pre.bias_correction()
r_ind,g_ind=pre.resize_crop()
pre.intensity_normalization()





















