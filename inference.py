# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 06:58:26 2021

@author: Administrator
"""


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import segmentation_models as sm
focal_loss = sm.losses.cce_dice_loss
from nilearn.image import resample_img
import SimpleITK as sitk
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from nilearn import image as nii
from nilearn import plotting


from tensorflow.keras.models import Model, load_model
import segmentation_models as sm
from segmentation_models.metrics import iou_score
focal_loss = sm.losses.cce_dice_loss
#from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json



##Preprocessing on test Images
def test_preprocessing(img_path):
    #applying bias correction
    img = sitk.ReadImage(img_path)
    img_mask = sitk.OtsuThreshold(img)
    img = sitk.Cast(img, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    img_c = corrector.Execute(img, img_mask)
    outputImageFileName = preprocessed_image
    sitk.WriteImage(img_c, outputImageFileName)

    #resizing and cropping
    target_shape = np.array((96,128,160))                   #reducing size of image from 256*256*192 to 96*128*80
    new_resolution = [2,]*3
    new_affine = np.zeros((4,4))
    new_affine[:3,:3] = np.diag(new_resolution)
    new_affine[:3,3] = target_shape*new_resolution/2.*-1
    new_affine[3,3] = 1.
    downsampled_and_cropped_nii = resample_img(preprocessed_image, target_affine=new_affine, target_shape=target_shape, interpolation='nearest')
    downsampled_and_cropped_nii.to_filename(preprocessed_image)
    image = sitk.ReadImage(preprocessed_image)

    #intensity normalizing
    rescaleFilter = sitk.RescaleIntensityImageFilter()
    rescaleFilter.SetOutputMaximum(255)
    rescaleFilter.SetOutputMinimum(0)
    image = rescaleFilter.Execute(image)
    sitk.WriteImage(image,preprocessed_image)

##Saving preprocessed test image
preprocessed_image = r'C:\Users\Administrator\Downloads\nfbs\T1w_MRI_test_data\T1Img\sub-01\preprocessed_test_image.nii.gz'
test_preprocessing(r'C:\Users\Administrator\Downloads\nfbs\T1w_MRI_test_data\T1Img\sub-01\T1w.nii.gz')


##loading model
model_architecture = r'C:\Users\Administrator\Downloads\nfbs\model_json.json'
with open(model_architecture, 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)

# load weights into model
loaded_model.load_weights(r"C:\Users\Administrator\Downloads\nfbs\model_weights_unet.h5")

#getting predictions 
orig_img=nib.load(preprocessed_image).get_data()
orig_img=np.expand_dims(orig_img,-1)
orig_img=np.expand_dims(orig_img,0)

pred_img=loaded_model.predict(orig_img)
pred_img=np.squeeze(pred_img)
orig_img=nib.load(preprocessed_image).get_data()

#converting prediction to nifti file
func = nib.load(preprocessed_image)
ni_img = nib.Nifti1Image(pred_img, func.affine)
nib.save(ni_img, r'C:\Users\Administrator\Downloads\nfbs\T1w_MRI_test_data\T1Img\sub-01\output_T1w_brain_mask.nii.gz')
pred_img=nib.load(r'C:\Users\Administrator\Downloads\nfbs\T1w_MRI_test_data\T1Img\sub-01\output_T1w_brain_mask.nii.gz')

#creating binary mask and stripping from raw image
pred_mask = math_img('img > 0.5', img=pred_img)
crop=pred_mask.get_data()*orig_img

#plotting outputs
pred_img=nib.load(r'C:\Users\Administrator\Downloads\nfbs\T1w_MRI_test_data\T1Img\sub-01\output_T1w_brain_mask.nii.gz').get_data()
fig,ax=plt.subplots(1,3,figsize=(15,10))
ax[0].set_title('Original image (cropped)')
ax[0].imshow(orig_img[orig_img.shape[0]//2])
ax[1].set_title('Predicted image')
ax[1].imshow(pred_img[pred_img.shape[0]//2])
ax[2].set_title('Skull stripped image')
ax[2].imshow(crop[crop.shape[0]//2])

#converting skull stripped to nifti file
ni_img = nib.Nifti1Image(crop, func.affine)
nib.save(ni_img, r'C:\Users\Administrator\Downloads\nfbs\output_T1w_brain.nii.gz')






