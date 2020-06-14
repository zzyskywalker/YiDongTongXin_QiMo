# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 21:23:49 2020

@author: 33578
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
snr=10
filenames=os.listdir("data/")
print(filenames)
windowsize=20
for file in filenames:
    
    dataframe=pd.read_csv("data/"+file)
    data_array=dataframe.to_numpy()
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(dataframe.to_numpy())
    data_mean=np.mean(data_array)
    data_array2=data_array-data_mean#减去直流量
    data_guiyihua=data_array2/np.math.sqrt(np.mean(abs(data_array2)**2));#能量归一化
    plt.subplot(2,1,2)
    plt.plot(data_guiyihua)
    result_list=[]
    for i in range(data_guiyihua.shape[0]//windowsize):

        s_noise= data_guiyihua[i*windowsize:i*windowsize+windowsize]
        signalpow = np.mean(abs(s_noise)**2)#信号功率
        noisepow = signalpow/(10**(snr/10))#噪声功率
        C20_hat = np.mean(s_noise**2)
        C21_hat = np.mean(abs(s_noise)**2)
        C21_hat = C21_hat-noisepow;#计算信号二阶累积量C21时，由于C21为信号模的平方
                                   #而我们接收的s是在AWGN信道下接收的，所以求C21时还应考虑噪声功率。
        C40_hat = np.mean(s_noise**4)-3*(C20_hat**2)
        C42_hat = np.mean(abs(s_noise)**4)-abs(C20_hat)**2-2*C21_hat**2
        
        C40_normal = C40_hat/(C21_hat**2)
        result_list.append([C20_hat,C21_hat,C40_hat,C42_hat,C40_normal])
        """
        cA5, cD5, cD4, cD3, cD2, cD1 =pywt.wavedec(s_noise, 'db10', level=5)
        ener_cA5 = np.square(cA5).sum()
        ener_cD5 = np.square(cD5).sum()
        ener_cD4 = np.square(cD4).sum()
        ener_cD3 = np.square(cD3).sum()
        ener_cD2 = np.square(cD2).sum()
        ener_cD1 = np.square(cD1).sum()
        ener = ener_cA5 + ener_cD1 + ener_cD2 + ener_cD3 + ener_cD4 + ener_cD5
        ratio_cA5 = ener_cA5/ener
        ratio_cD5 = ener_cD5/ener
        ratio_cD4 = ener_cD4/ener
        ratio_cD3 = ener_cD3/ener
        ratio_cD2 = ener_cD2/ener
        ratio_cD1 = ener_cD1/ener
        result_list.append([ener_cA5,ener_cD1,ener_cD2,ener_cD3,ener_cD4,ener_cD5,ratio_cA5,ratio_cD1,ratio_cD2,ratio_cD3,ratio_cD4,ratio_cD5])
    """
    print(file)
    
    #result_dataframe=pd.DataFrame(result_list,columns=["ener_cA5","ener_cD1","ener_cD2","ener_cD3","ener_cD4","ener_cD5","ratio_cA5","ratio_cD1","ratio_cD2","ratio_cD3","ratio_cD4","ratio_cD5"])
    result_dataframe=pd.DataFrame(result_list,columns=["C20_hat","C21_hat","C40_hat","C42_hat","C40_normal"])
    plt.figure()
    plt.plot(result_dataframe.loc[:,"C40_normal"])
   
    print(result_dataframe.shape)
    result_dataframe.to_csv("feature/"+file,index=False)