# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 14:38:25 2023

@author: pedro
"""

import torch
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from PIL import Image

# ------------------ Dataset Class -----------------

class Dataset(Dataset):
    
    def __init__(self, csv_file, data_dir,img_dir, transform=None):
        #Image directory
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.transform = transform
        data_dircsv_file = os.path.join(self.data_dir, csv_file)
        #Load the csv that contains image info
        self.data_name = pd.read_csv(data_dircsv_file)
        self.len = self.data_name.shape[0]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # Image file path
        img_name = os.path.join(
            self.data_dir,
            self.img_dir,
            self.data_name.iloc[index, 1])
        # Open image file
        image = Image.open(img_name)
        
        # Class label
        label = self.data_name.iloc[index, 2]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
# Directories
# look for current directory

#os.getcwd()
# The directories are going to be different in your case c:
csv_file = 'metadata.csv'
directory = 'C:\\Users\\*\\OneDrive\\Escritorio\\Kaggle_MLandAI\\Brain_Tumor_Dataset'
img_dir_Tumor = 'C:\\Users\\*\\OneDrive\\Escritorio\\Kaggle_MLandAI\\Brain_Tumor_Dataset\\Brain Tumor Data Set\\Brain Tumor Data Set\\Brain Tumor' 
img_dir_NoTumor = 'C:\\Users\\*\\OneDrive\\Escritorio\\Kaggle_MLandAI\\Brain_Tumor_Dataset\\Brain Tumor Data Set\\Brain Tumor Data Set\\Healthy'       

dataset_Brain_Tumor = Dataset(
    csv_file=csv_file,
    data_dir=directory,
    img_dir=img_dir_Tumor
    )

len(dataset_Brain_Tumor)

for i in range(5):
    image = dataset_Brain_Tumor[i][0]
    label = dataset_Brain_Tumor[0][1] # the label is going to be the same cause we are only extracting Tumor dataset
    plt.imshow(image, cmap='gray')
    plt.title(label)


