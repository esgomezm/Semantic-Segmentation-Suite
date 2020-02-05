# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:21:06 2020

@author: biig
"""

#Change to rgb
#PARA GOOGLE COLAB

#Install all the necesarry packs
!pip install SimpleITK
import numpy as np
from skimage import io
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import os.path
from os import path
import re
import glob


from google.colab import drive
drive.mount('/content/drive')


path2data = r'/content/drive/My Drive/TFG/Seg_github/data/'
path2rgbdata= r'/content/drive/My Drive/TFG/Seg_github/RGB_data/'
names=['test','test_labels','train','train_labels','val','val_labels']
names_length = len(names)

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    return numpyImage

def to_rgb(im):
    n, w, h = im.shape
    ret = np.empty((n, w, h, 3), dtype=np.uint8)
    ret[:, :, :, 2] =  ret[:, :, :, 1] =  ret[:, :, :, 0] =  im
    return ret


for i in pbar(range(names_length)):
    if not path.exists(path2rgbdata + names[i] + '/' ):
        os.mkdir(path2rgbdata + names[i] + '/')
    for file in os.listdir(path2data + names[i]):
        old_pic=load_itk_image(path2data + names[i] + '/' + file)
        n, w, h = old_pic.shape
        filename=os.path.splitext(file)[0]
        if '_L.tif' in file: 
            old_pic[old_pic>0]=255           
            im8 = old_pic.astype('uint8')
        else:
            old_pic=load_itk_image(path2data + names[i] + '/' + file)            
            im8 = (old_pic/256).astype('uint8')
        rgbim=to_rgb(im8)
        for j in range(n):
            im2save=rgbim[j,:,:,:]
            imsitk = sitk.GetImageFromArray(im2save.astype(np.uint8))
            sitk.WriteImage(imsitk, path2rgbdata + names[i] +'/' +  filename + '_' + str(j) + '.tif')