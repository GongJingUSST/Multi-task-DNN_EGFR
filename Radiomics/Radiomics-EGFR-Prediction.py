# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 15:41:42 2022

@author: DELL
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
from sklearn import svm, linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,auc,confusion_matrix
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score,matthews_corrcoef
from sklearn.model_selection import LeaveOneOut
from sklearn.svm import SVC,LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectPercentile, f_classif, chi2, SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas import DataFrame as DF
from imblearn.over_sampling import SMOTE
import xlrd
import seaborn as sns


def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

def confindence_interval_compute(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
#        indices = rng.random_integers(0, len(y_pred), len(y_pred))
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
    
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_std = sorted_scores.std()
    
    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
    return confidence_lower,confidence_upper,confidence_std

def prediction_score(truth, predicted):
    TN, FP, FN, TP = confusion_matrix(truth, predicted, labels=[0,1]).ravel()
    print(TN, FP, FN, TP)
    ACC = (TP+TN)/(TN+FP+FN+TP)
    SEN = TP/(FN+TP)
    SPE = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)
    print('ACC:',ACC)
    print('Sensitivity:',SEN)
    print('Specifity:',SPE)
    print('PPV:',PPV)
    print('NPV:',NPV)
    OR = (TP*TN)/(FP*FN)
    print('OR:',OR)
    F1_3 = f1_score(truth, predicted)
    print('F1:', F1_3)
    F1_w3 = f1_score(truth, predicted,average='weighted')
    print('F1_weight:',F1_w3)
    MCC3 = matthews_corrcoef(truth, predicted)
    print('MCC:',MCC3)
    
if __name__ == '__main__': 
    lw = 1.5
    font = {'family' : 'Arial',
            'weight' :  'medium',
            'size'   : 12,}
    plt.rc('font', **font)
    Train_File = open('../Result/TrainData_Radiomics_Feature.csv')
    Train_Data = pd.read_csv(Train_File)
    Train_Data = Train_Data.fillna('0')
    Train_PatientID = list(Train_Data['PatientID'])
    Train_EGFR = np.array(Train_Data['EGFR Status'])
    Train_Feature = Train_Data.values[:,:-2]
    FeatureName = list(Train_Data.head(0))[:-2]
    
    Valid_File = open('../Result/VD1_Radiomics_Feature.csv')
    Valid_Data = pd.read_csv(Valid_File)
    Valid_Data = Valid_Data.fillna('0')
    Valid_PatientID = list(Valid_Data['PatientID'])
    Valid_EGFR = np.array(Valid_Data['EGFR Status'])
    Valid_Feature = Valid_Data.values[:,:-2]
    
    TCIA_File = open('../Result/TCIA_Radiomics_Feature.csv')
    TCIA_Data = pd.read_csv(TCIA_File)
    TCIA_Data = TCIA_Data.fillna('0')
    TCIA_PatientID = list(TCIA_Data['PatientID'])
    TCIA_EGFR = np.array(TCIA_Data['EGFR Status'])
    TCIA_Feature = TCIA_Data.values[:,:-2]
    
    scaler = MinMaxScaler(feature_range=(0,1))
    Train_Feature = scaler.fit_transform(Train_Feature)
    Valid_Feature = scaler.transform(Valid_Feature)
    TCIA_Feature = scaler.transform(TCIA_Feature)

    clf = linear_model.Lasso(alpha=1,random_state=0)#5
    
    selector = RFE(clf, n_features_to_select=5, step=5).fit(Train_Feature, Train_EGFR)
    transformed_train = selector.transform(Train_Feature)   
    transformed_valid = selector.transform(Valid_Feature)
    transformed_TCIA = selector.transform(TCIA_Feature)
    
    indices = list(np.where(selector.support_==True)[0])
    print(np.array(FeatureName)[indices])
    
    x_train, y_train = SMOTE(sampling_strategy='auto',random_state=0).fit_resample(transformed_train, Train_EGFR)
    clf_EGFR = SVC(kernel="rbf", probability=True, random_state=0)
    
    clf_EGFR.fit(x_train, y_train)
    
    valid_prob = clf_EGFR.predict_proba(transformed_valid)[:,1]
    valid_pred_label = clf_EGFR.predict(transformed_valid)
    valid_fpr,valid_tpr,valid_threshold = roc_curve(Valid_EGFR, np.array(valid_prob)) ###计算真正率和假正率
    valid_auc = auc(valid_fpr,valid_tpr)
    valid_l, valid_h, valid_std = confindence_interval_compute(np.array(valid_prob), np.array(Valid_EGFR))
    print('validation Dataset AUC:%.2f'%valid_auc,'+/-%.2f'%valid_std,'  95% CI:[','%.2f,'%valid_l,'%.2f'%valid_h,']')
    print('validation Dataset ACC:%.4f'%accuracy_score(Valid_EGFR,valid_pred_label)) 
    prediction_score(Valid_EGFR,valid_pred_label)
    
    TCIA_prob = clf_EGFR.predict_proba(transformed_TCIA)[:,1]
    TCIA_pred_label = clf_EGFR.predict(transformed_TCIA)
    TCIA_fpr,TCIA_tpr,TCIA_threshold = roc_curve(TCIA_EGFR, np.array(TCIA_prob)) ###计算真正率和假正率
    TCIA_auc = auc(TCIA_fpr,TCIA_tpr)
    TCIA_l, TCIA_h, TCIA_std = confindence_interval_compute(np.array(TCIA_prob), np.array(TCIA_EGFR))
    print('TCIA Dataset AUC:%.2f'%TCIA_auc,'+/-%.2f'%TCIA_std,'  95% CI:[','%.2f,'%TCIA_l,'%.2f'%TCIA_h,']')
    print('TCIA Dataset ACC:%.4f'%accuracy_score(TCIA_EGFR,TCIA_pred_label)) 
    prediction_score(TCIA_EGFR,TCIA_pred_label)
    print('----------------------------------------------')