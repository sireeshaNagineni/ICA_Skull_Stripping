# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:48:10 2021

@author: Administrator
"""

import os
from sklearn.model_selection import train_test_split
import numpy as np
import random
import nibabel as nib

import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import concatenate,Dropout
from tensorflow.keras.layers import Multiply, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization, Flatten, Conv2D, AveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
from segmentation_models.metrics import iou_score
focal_loss = sm.losses.cce_dice_loss
#from keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


mask_path = r'C:\Users\Administrator\Downloads\nfbs\resized\mask'
raw_path = r'C:\Users\Administrator\Downloads\nfbs\resized\raw'

resized_mask_temp = os.listdir(mask_path)
resized_img_temp = os.listdir(raw_path)
resized_img = []
resized_mask = []

for i in resized_mask_temp:
    resized_mask.append(mask_path + '/'+i)
for i in resized_img_temp:
    resized_img.append(raw_path + '/'+i)   
    
    

def split(resized_img,resized_mask): 
    X_train,X_test,y_train,y_test=train_test_split(resized_img,resized_mask,test_size=0.1)
    return X_train,X_test,y_train,y_test

def data_gen(img_list, mask_list, batch_size):
    '''Custom data generator to feed image to model'''
    c = 0
    n = [i for i in range(len(img_list))]  #List of training images
    random.shuffle(n)
    
    while (True):
      img = np.zeros((batch_size, 96, 128, 160,1)).astype('float')   #adding extra dimensions as conv3d takes file of size 5
      mask = np.zeros((batch_size, 96, 128, 160,1)).astype('float')

      for i in range(c, c+batch_size): 
        train_img = nib.load(img_list[n[i]]).get_data()
        
        train_img=np.expand_dims(train_img,-1)
        train_mask = nib.load(mask_list[n[i]]).get_data()

        train_mask=np.expand_dims(train_mask,-1)

        img[i-c]=train_img
        mask[i-c] = train_mask
      c+=batch_size
      if(c+batch_size>=len(img_list)):
        c=0
        random.shuffle(n)

      yield img,mask

def convolutional_block(input, filters=3, kernel_size=3, batchnorm = True):
    '''conv layer followed by batchnormalization'''
    x = Conv3D(filters = filters, kernel_size = (kernel_size, kernel_size,kernel_size),
               kernel_initializer = 'he_normal', padding = 'same')(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(filters = filters, kernel_size = (kernel_size, kernel_size,kernel_size),
               kernel_initializer = 'he_normal', padding = 'same')(input)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    return x
def resunet_opt(input_img, filters = 64, dropout = 0.2, batchnorm = True):
    
    """Residual Unet + Dense Atrous convolution + Rmp block"""
    conv1 = convolutional_block(input_img, filters * 1, kernel_size = 3, batchnorm = batchnorm)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)
    drop1 = Dropout(dropout)(pool1)
    
    conv2 = convolutional_block(drop1, filters * 2, kernel_size = 3, batchnorm = batchnorm)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)
    drop2 = Dropout(dropout)(pool2)
    
    conv3 = convolutional_block(drop2, filters * 4, kernel_size = 3, batchnorm = batchnorm)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)
    drop3 = Dropout(dropout)(pool3)
    
    conv4 = convolutional_block(drop3, filters * 8, kernel_size = 3, batchnorm = batchnorm)
    pool4 = MaxPooling3D((2, 2, 2))(conv4)
    drop4 = Dropout(dropout)(pool4)
    
    conv5 = convolutional_block(drop4, filters = filters * 16, kernel_size = 3, batchnorm = batchnorm)
    conv5 = convolutional_block(conv5, filters = filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    ups6 = Conv3DTranspose(filters * 8, (3, 3, 3), strides = (2, 2, 2), padding = 'same',activation='relu',kernel_initializer='he_normal')(conv5)
    ups6 = concatenate([ups6, conv4])
    ups6 = Dropout(dropout)(ups6)
    conv6 = convolutional_block(ups6, filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    ups7 = Conv3DTranspose(filters * 4, (3, 3, 3), strides = (2, 2, 2), padding = 'same',activation='relu',kernel_initializer='he_normal')(conv6)
    ups7 = concatenate([ups7, conv3])
    ups7 = Dropout(dropout)(ups7)
    conv7 = convolutional_block(ups7, filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    ups8 = Conv3DTranspose(filters * 2, (3, 3, 3), strides = (2, 2, 2), padding = 'same',activation='relu',kernel_initializer='he_normal')(conv7)
    ups8 = concatenate([ups8, conv2])
    ups8 = Dropout(dropout)(ups8)
    conv8 = convolutional_block(ups8, filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    ups9 = Conv3DTranspose(filters * 1, (3, 3, 3), strides = (2, 2, 2), padding = 'same',activation='relu',kernel_initializer='he_normal')(conv8)
    ups9 = concatenate([ups9, conv1])
    ups9 = Dropout(dropout)(ups9)
    conv9 = convolutional_block(ups9, filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv3D(1, (1, 1, 2), activation='sigmoid',padding='same')(conv9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
      
     

im_height=96
im_width=128
img_depth=160
epochs=60
X_train,X_test,y_train,y_test = split(resized_img,resized_mask)
train_gen = data_gen(X_train,y_train, batch_size = 4)
val_gen = data_gen(X_test,y_test, batch_size = 4)
channels=1
input_img = Input((im_height, im_width,img_depth,channels), name='img')
model = resunet_opt(input_img, filters=16, dropout=0.05, batchnorm=True)
model.summary()
model.compile(optimizer=Adam(lr=1e-1),loss=focal_loss,metrics=[iou_score,'accuracy'])
#fitting the model
file_path = r'C:\Users\Administrator\Downloads\nfbs\model_weights.h5'

callbacks=callbacks = [
        ModelCheckpoint(file_path, verbose=1, save_best_only=True, save_weights_only=False)]
result=model.fit(train_gen,steps_per_epoch=16,epochs=epochs,validation_data=val_gen,validation_steps=16,initial_epoch=0,callbacks=callbacks)
model_json = model.to_json()
json_path = r'C:\Users\Administrator\Downloads\nfbs\model_json.json'

file_path1 = r'C:\Users\Administrator\Downloads\nfbs\model_weights_unet.h5'
with open(json_path, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(file_path1)
    
 #'best_model.h5'   
    
    
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    