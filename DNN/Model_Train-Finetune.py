# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 15:26:28 2022

@author: DELL
"""


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['CUDA_VISBLE_DEVICES'] = "0,1"
from monai.utils import set_determinism, first
from monai.transforms import (LoadImaged,
                              Lambda,
                              BorderPadd,
                              SpatialPadd,
                              SpatialCropd,
                              ScaleIntensityRanged,
                              AddChanneld,
                              Spacingd,
                              CenterSpatialCropd,
                              Compose,
                              RandShiftIntensityd,
                              RandFlipd,
                              RandRotate90d,
                              EnsureTyped,
                              AsDiscrete,
                              EnsureType,
                              Activations,
                              )
from monai.data import CacheDataset,  decollate_batch, Dataset
from monai.networks.nets import ResNet, DenseNet121
from monai.networks.layers import Norm
from monai.metrics import ROCAUCMetric
from monai.losses import FocalLoss, DiceLoss
from monai.networks.blocks import  UnetResBlock, UnetUpBlock, UnetBasicBlock, UnetOutBlock, ResidualUnit,MaxAvgPool
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import Norm
from monai.networks.nets import ViT
from monai.metrics import DiceMetric
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import Module
import xlrd
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc

class MultiTaskModel(Module):
    def __init__(self, spatial_dims: int = 3,
        in_channels: int = 1,
        seg_channels: int = 1,
        clf_channels: int = 2,
        dropout_prob: float = 0.5,):
        super().__init__()
        self.in_tr16 = ResidualUnit(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=16)
        self.down_tr32 = UnetResBlock(spatial_dims=spatial_dims, in_channels=16, out_channels=32,
                                    kernel_size=3, stride=2, norm_name=Norm.BATCH, dropout=dropout_prob)
        self.down_tr64 = UnetResBlock(spatial_dims=spatial_dims, in_channels=32, out_channels=64,
                                    kernel_size=3, stride=2, norm_name=Norm.BATCH, dropout=dropout_prob)
        self.down_tr128 = UnetResBlock(spatial_dims=spatial_dims, in_channels=64, out_channels=128,
                                    kernel_size=3, stride=2, norm_name=Norm.BATCH, dropout=dropout_prob)
        self.down_tr256 = UnetResBlock(spatial_dims=spatial_dims, in_channels=128, out_channels=256,
                                    kernel_size=3, stride=2, norm_name=Norm.BATCH, dropout=dropout_prob)
        self.down_tr512 = UnetResBlock(spatial_dims=spatial_dims, in_channels=256, out_channels=512,
                                    kernel_size=3, stride=2, norm_name=Norm.BATCH, dropout=dropout_prob)
        self.maxavgpool = MaxAvgPool(spatial_dims=spatial_dims, kernel_size=3)
        self.clf_model = nn.Linear(in_features=1024, out_features=2)
    def forward(self, x):
        out16 = self.in_tr16(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out512 = self.down_tr512(out256)
        out = self.maxavgpool(out512)
        clf_result = self.clf_model(out.view(-1,1024))
        return  clf_result
    
if __name__ == '__main__':
    img_path = r'..\Img-NII'
    
    data = xlrd.open_workbook(r'..\..\PatientList-All.xls')
    table = data.sheets()[0]
    PatientID_all = table.col_values(0)[1:]
    PatientID = [str(int(i)) for i in PatientID_all]
    X_Pos = np.array(table.col_values(2)[1:])
    Y_Pos = np.array(table.col_values(3)[1:])
    Z_Pos = np.array(table.col_values(4)[1:])
    Center_Pos = [(int(X_Pos[i]), int(Y_Pos[i]), int(Z_Pos[i])) for i in range(len(X_Pos))]
    Label = np.array(table.col_values(7)[1:])
    Label = [1 if i == '是' else 0 for i in Label]
    images = [os.path.join(img_path, str(int(i))+'.nii.gz') for i in PatientID]
    data_dicts = [
        {"image": image_name, 'roi_center': roi_center, "label": label_name}
        for image_name,  roi_center, label_name in zip(images, Center_Pos, Label)
    ]
    index = [i for i in range(len(data_dicts))]
    np.random.seed(0)
    np.random.shuffle(index)
    data_dicts = np.array(data_dicts)[index]
    train_files, val_files = data_dicts[:-164], data_dicts[-164:]

    set_determinism(seed=0)
    
    SpatialPaddLambda = Lambda(func=lambda d:BorderPadd(keys=['image'], 
                              spatial_border=(0, 0, 0, 0,
                                      max(64-d['roi_center'][2],0), max(0, d['roi_center'][2]+64-d['image'].shape[3])))(d)
                              )
    SpatialCropLamda = Lambda(
                       lambda d:SpatialCropd(keys=['image'],roi_center=d['roi_center'],
                                             roi_size=(128,128,128))(d)
                             )
    train_transformer = Compose([LoadImaged(keys=['image'], dtype=np.int16),
                         AddChanneld(keys=['image']),
                         ScaleIntensityRanged(keys='image',a_min=-1200,a_max=400, b_min=0, b_max=1),
                         SpatialPaddLambda,
                         SpatialCropLamda,
                         Spacingd(keys=['image'], pixdim=(1, 1, 1)),
                         CenterSpatialCropd(keys=['image'],roi_size=(96,96,96)),
                         SpatialPadd(keys=['image'],spatial_size=(96,96,96)),
                         RandFlipd(keys=['image', 'mask'], spatial_axis=[0], prob=0.1),
                         RandFlipd(keys=['image', 'mask'], spatial_axis=[1], prob=0.1),
                         RandFlipd(keys=['image', 'mask'], spatial_axis=[2], prob=0.1),
                         RandRotate90d(keys=['image'], max_k=3, prob=0.5),
                         RandShiftIntensityd(keys=['image'], offsets=0.5, prob=0.1),
                         EnsureTyped(keys=['image']),])

    valid_transformer = Compose([LoadImaged(keys=['image'], dtype=np.int16),
                         AddChanneld(keys=['image']),
                         ScaleIntensityRanged(keys='image',a_min=-1200,a_max=400, b_min=0, b_max=1),
                         SpatialPaddLambda,
                         SpatialCropLamda,
                         Spacingd(keys=['image'],pixdim=(1,1,1)),
                         CenterSpatialCropd(keys=['image'],roi_size=(96,96,96)),
                         SpatialPadd(keys=['image'], spatial_size=(96, 96, 96)),
                         EnsureTyped(keys=['image']),])
    

    train_ds = CacheDataset(data=train_files, transform=train_transformer, cache_rate=1, num_workers=0)
    train_loader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    valid_ds = CacheDataset(data=val_files, transform=valid_transformer, cache_rate=1, num_workers=0)
    val_loader = DataLoader(dataset=valid_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskModel()

    model = ViT(in_channels=1, img_size=(96,96,96),patch_size=(16,16,16),  pos_embed='conv', classification=True).to(device)
    model_dict = model.state_dict()
    save_model = torch.load("best_metric_model_Multi-Task.pth")
    state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)

    for k,v in model.named_parameters():
        if k not in ["clf_model.weight","clf_model.bias"]:
            v.requires_grad=False
            
    if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            model = model.cuda()        
    clf_loss = FocalLoss(to_onehot_y=True, weight=[530/288, 1], gamma=2)

    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = lr_scheduler.StepLR(optim, step_size=100, gamma=0.1)

    post_pred = Compose([EnsureType(), Activations(softmax=True)])
    seg_pred = Compose([EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    auc_metric = ROCAUCMetric()
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    metric_values = []
    epochs = 200
    # use amp to accelerate training
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True
    writer = SummaryWriter()
    for epoch in range(epochs):
        print("*" * 15)
        print(f"epoch {epoch+1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (batch_data['image'].to(device),
                              batch_data['label'].to(device),)
            optim.zero_grad()
            with torch.cuda.amp.autocast():
                clf_outputs = model(inputs)#
                loss = clf_loss(clf_outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train loss: {loss.item():.4f}")
            writer.add_scalar("train loss", loss.item(), epoch*epoch_len+step)
        scheduler.step()
        epoch_loss /= step
        print(f"epoch {epoch+1} average loss: {epoch_loss:.4f}")


        if (epoch+1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                test_loss = []
                test_label = []
                test_prob = []
                for val_data in val_loader:
                    val_images, val_labels = val_data['image'].cuda(),val_data['label'].cuda()
                    test_label.append(val_labels.cpu().numpy())
                    val_clf_outputs = model(val_images)#
                    output = torch.softmax(val_clf_outputs, dim = 1)
                    test_prob.append(output.cpu().numpy()[0][1])
                fpr,tpr,threshold = roc_curve(test_label, np.array(test_prob)) ###计算真正率和假正率
                auc_result = auc(fpr,tpr)
                acc_metric = accuracy_score(np.array(test_label),(np.array(test_prob)>0.5).astype(int))*100

                if acc_metric >best_metric:
                    best_metric = acc_metric
                    best_metric_epoch = epoch+1
                    torch.save(model.module.state_dict(), "best_metric_Multi-task_model.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f}\nbest accuracy: {:.4f} at epoch {}".format(
                        epoch+1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_accuracy", acc_metric, epoch+1)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()