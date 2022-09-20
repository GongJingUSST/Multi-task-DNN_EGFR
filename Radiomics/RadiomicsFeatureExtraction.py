# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:16:38 2022

@author: DELL
"""


import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import os
from pandas import DataFrame as DF
import time
from tqdm import tqdm


def Extract_Features(image,mask):
    paramsFile = os.path.abspath('params.yaml')
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    result = extractor.execute(image, mask)
    general_info = {'diagnostics_Configuration_EnabledImageTypes','diagnostics_Configuration_Settings',
                    'diagnostics_Image-interpolated_Maximum','diagnostics_Image-interpolated_Mean',
                    'diagnostics_Image-interpolated_Minimum','diagnostics_Image-interpolated_Size',
                    'diagnostics_Image-interpolated_Spacing','diagnostics_Image-original_Hash',
                    'diagnostics_Image-original_Maximum','diagnostics_Image-original_Mean',
                    'diagnostics_Image-original_Minimum','diagnostics_Image-original_Size',
                    'diagnostics_Image-original_Spacing','diagnostics_Mask-interpolated_BoundingBox',
                    'diagnostics_Mask-interpolated_CenterOfMass','diagnostics_Mask-interpolated_CenterOfMassIndex',
                    'diagnostics_Mask-interpolated_Maximum','diagnostics_Mask-interpolated_Mean',
                    'diagnostics_Mask-interpolated_Minimum','diagnostics_Mask-interpolated_Size',
                    'diagnostics_Mask-interpolated_Spacing','diagnostics_Mask-interpolated_VolumeNum',
                    'diagnostics_Mask-interpolated_VoxelNum','diagnostics_Mask-original_BoundingBox',
                    'diagnostics_Mask-original_CenterOfMass','diagnostics_Mask-original_CenterOfMassIndex',
                    'diagnostics_Mask-original_Hash','diagnostics_Mask-original_Size',
                    'diagnostics_Mask-original_Spacing','diagnostics_Mask-original_VolumeNum',
                    'diagnostics_Mask-original_VoxelNum','diagnostics_Versions_Numpy',
                    'diagnostics_Versions_PyRadiomics','diagnostics_Versions_PyWavelet',
                    'diagnostics_Versions_Python','diagnostics_Versions_SimpleITK',
                    'diagnostics_Image-original_Dimensionality'}
    features = dict((key, value) for key, value in result.items() if key not in general_info)
    feature_info = dict((key, value) for key, value in result.items() if key in general_info)
    return features,feature_info

if __name__ == '__main__':
    train_img_path = r'..\TrainData\Image'
    train_mask_path = r'..\TrainData\Mask'
    train_list = pd.read_csv(r'..\TrainData\PatientList.csv')
    train_PatientID = np.array(train_list['PatientID'])
    train_EGFR = np.array(train_list['EGFR Status'])
    train_images = [os.path.join(train_img_path, str(i)+'.nii.gz') for i in train_PatientID]
    train_masks = [os.path.join(train_mask_path, str(i)+'.nii.gz') for i in train_PatientID]
    train_files = [
        {"image": image_name, "mask":mask_name, "label": label_name, "ID": ID}
        for image_name, mask_name, label_name, ID in zip(train_images, train_masks, train_EGFR, train_PatientID)
    ]
    start = time.perf_counter()
    train_Feature = []
    for train_file in tqdm(train_files):
        Image = sitk.ReadImage(train_file['image'])
        Mask = sitk.ReadImage(train_file['mask'])
        Mask.CopyInformation(Image)
        feature, feature_info = Extract_Features(Image, Mask) 
        feature['PatientID'] = train_file["ID"]
        feature['EGFR Status'] = train_file["label"]    
        train_Feature.append(feature)
    df = DF(train_Feature).fillna('0')
    df.to_csv('../Result/TrainData_Radiomics_Feature.csv', index = False, sep=',')
    end = time.perf_counter()
    print(end-start)
    
    valid_img_path = r'..\Result\VD1\Image'
    valid_mask_path = r'..\Result\VD1\Mask'
    valid_list = pd.read_csv(r'..\Result\VD1\PatientList.csv')
    valid_PatientID = np.array(valid_list['PatientID'])
    valid_EGFR = np.array(valid_list['EGFR Status'])
    valid_images = [os.path.join(valid_img_path, str(i)+'.nii.gz') for i in valid_PatientID]
    valid_masks = [os.path.join(valid_mask_path, str(i)+'.nii.gz') for i in valid_PatientID]
    valid_files = [
        {"image": image_name, "mask":mask_name, "label": label_name, "ID": ID}
        for image_name, mask_name, label_name, ID in zip(valid_images, valid_masks, valid_EGFR, valid_PatientID)
    ]
    start = time.perf_counter()
    valid_Feature = []
    for valid_file in tqdm(valid_files):
        Image = sitk.ReadImage(valid_file['image'])
        Mask = sitk.ReadImage(valid_file['mask'])
        Mask.CopyInformation(Image)
        feature, feature_info = Extract_Features(Image, Mask) 
        feature['PatientID'] = valid_file["ID"]
        feature['EGFR Status'] = valid_file["label"]    
        valid_Feature.append(feature)
    df = DF(valid_Feature).fillna('0')
    df.to_csv('../Result/VD1_Radiomics_Feature.csv', index = False, sep=',')
    end = time.perf_counter()
    print(end-start)
    
    TCIA_img_path = r'F:\LungCancer\EGFR-GGN-Prediction\Result\VD2\Image'
    TCIA_mask_path = r'F:\LungCancer\EGFR-GGN-Prediction\Result\VD2\Mask'    
    TCIA_list = pd.read_csv(r'F:\LungCancer\EGFR-GGN-Prediction\Result\VD2\PatientList.csv')
    TCIA_PatientID = np.array(TCIA_list['PatientID'])
    TCIA_EGFR = np.array(TCIA_list['EGFR Status'])
    TCIA_images = [os.path.join(TCIA_img_path, str(i)+'.nii.gz') for i in TCIA_PatientID]
    TCIA_masks = [os.path.join(TCIA_mask_path, str(i)+'.nii.gz') for i in TCIA_PatientID]
    TCIA_files = [
        {"image": image_name, "mask":mask_name, "label": label_name, "ID": ID}
        for image_name, mask_name, label_name, ID in zip(TCIA_images, TCIA_masks, TCIA_EGFR, TCIA_PatientID)
    ]
    
    start = time.perf_counter()
    TCIA_Feature = []
    for TCIA_file in tqdm(TCIA_files):
        Image = sitk.ReadImage(TCIA_file['image'])
        Mask = sitk.ReadImage(TCIA_file['mask'])
        Mask.CopyInformation(Image)
        feature, feature_info = Extract_Features(Image, Mask) 
        feature['PatientID'] = TCIA_file["ID"]
        feature['EGFR Status'] = TCIA_file["label"]    
        TCIA_Feature.append(feature)
    df = DF(TCIA_Feature).fillna('0')
    df.to_csv('../Result/TCIA_Radiomics_Feature.csv', index = False, sep=',')
    end = time.perf_counter()
    print(end-start)