# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score,mean_absolute_error,explained_variance_score,mean_squared_error,median_absolute_error
from sklearn.model_selection import LeaveOneOut,LeavePOut
from sklearn.model_selection import ShuffleSplit,StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def mean_fun(onelist):
    count = 0
    for i in onelist:
        count += i
    return float(count/len(onelist))

def loadData(filename1,filename2):


    fr=open(filename1)
    fr2 = open(filename2)
    fileList=fr.readlines()
    fileList2 = fr2.readlines()
    X=[];y=[]
    X2=[];y2=[]
    m = len(fileList)
    m2 = len(fileList2)
    for i in range(m):
        strLine=fileList[i].split('\t')
        numFeat = len(strLine)-1
        lineList=[]
        for p in range(0,numFeat):
            lineList.append(float(strLine[p]))
        X.append(lineList)
        #print (X)
        y.append(float(strLine[-1]))

    for i in range(m2):
        strLine2 = fileList2[i].split('\t')
        numFeat2 = len(strLine2)-1
        lineList2 = []
        for p in range(0, numFeat2):
            lineList2.append(float(strLine2[p]))
        X2.append(lineList2)
        y2.append(float(strLine2[-1]))



    params = {'learning_rate': 0.01, 'n_estimators': 600, 'max_depth': 13, 'min_samples_leaf': 70,'min_samples_split': 2, 'max_features': 14, 'random_state': 10}
    clf = GradientBoostingClassifier(**params)
    clf.fit(X, y)
    proba = clf.predict_proba(X2)
    y_pred = clf.predict(X2)
    y2_proba=[]
    for i in range(len(proba)):
        y2_proba.append(proba[i][1])
    AUC = roc_auc_score(y2, y2_proba)
    print(AUC)
    fr = open('/ifs/gdata1/yangwenyi/xiaohei/result.txt', 'w')
    for i in range(len(y_pred)):
        fr.write(str(y_pred[i]) + '\t' + str(proba[i][1]) + '\n')
    fr.close()
    fpr,tpr,thresholds = roc_curve(y2, y2_proba)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)

'''
    EVS = explained_variance_score(y1, y2)  # 可释方差值
    print("EVS: %.4f" % EVS)
    MAE = mean_absolute_error(y1, y2)  # 平均绝对误差
    print("MAE: %.4f" % MAE)
    MSE = mean_squared_error(y1, y2)  # 均方误差
    print("MSE: %.4f" % MSE)
    MedAE = median_absolute_error(y1, y2)  # 中值绝对误差
    print("MedAE: %.4f" % MedAE)
    
    R2 = r2_score(y1, y2)  # R方值
    print("R2:%.4f" % R2)
'''

if __name__ == '__main__':
    filename1='/ifs/data1/liwenkai/hin2vec/hin2vec-master/allow_circle/two/1/train/train1.txt'
    filename2='/ifs/data1/liwenkai/hin2vec/hin2vec-master/allow_circle/two/1/test/test1.txt'
    loadData(filename1, filename2)

