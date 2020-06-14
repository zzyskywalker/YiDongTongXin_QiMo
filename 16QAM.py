# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 13:29:30 2020

@author: 33578
"""

from scipy import signal, special
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

T = 1               #基带信号宽度，也就是频率
nb = 800          #定义传输的比特数
delta_T = T/200     #采样间隔
fs = 1/delta_T      #采样频率
fc = 2/T           #载波频率
SNR = 20             #信噪比

t = np.arange(0, nb*T, delta_T)
N = len(t)

# 产生基带信号
data = [1 if x > 0.5 else 0 for x in np.random.randn(1, nb*20)[0]]  #调用随机函数产生任意在0到1的1*nb的矩阵，大于0.5显示为1，小于0.5显示为0
#print("data:",data)
data0 = []                             #创建一个1*nb/delta_T的零矩阵
for q in range(nb):
    data0 += [data[q]]*int(1/delta_T)  #将基带信号变换成对应波形信号
#plt.plot(data0)


idata = data[0:(nb-1):2]       #串并转换，将奇偶位分开，间隔为2，i是奇位 q是偶位
qdata = data[1:nb:2]       
print(len(idata),len(qdata))


idata_array=np.array(idata)                           #每两个比特放在一起
idata2=idata_array.reshape(int(len(idata)/2),2)  
qdata_array=np.array(qdata)
qdata2=qdata_array.reshape(int(len(qdata)/2),2)

idata3=[]
qdata3=[]
print(idata2.shape[0])      #将两个比特转化为十进制0，1，2，3
for i in range(idata2.shape[0]):
    idata3.append(int(''.join(str(j) for j in idata2[i]),2))
    qdata3.append(int(''.join(str(j) for j in qdata2[i]),2))
print(idata2)
print(idata3,len(idata3))

    #将基带信号转换成极性码,映射
idata4=np.array(idata3)*2-3
qdata4=np.array(qdata3)*2-3
print(idata4)


ich = []                          #创建一个1*nb/delta_T/2的零矩阵，以便后面存放奇偶位数据
qch = []   
for i in range(int(nb/4)):
    ich += [idata4[i]]*int(1/delta_T)    #奇位码元转换为对应的波形信号
    qch += [qdata4[i]]*int(1/delta_T)    #偶位码元转换为对应的波形信号
plt.figure()
plt.plot(ich)

a=[]
b=[]
for j in range(int(N/4)):
    a.append(np.math.sqrt(2/T)*np.math.cos(2*np.math.pi*fc*t[j]))    #余弦函数载波
    b.append(np.math.sqrt(2/T)*np.math.sin(2*np.math.pi*fc*t[j]))    #正弦函数载波
    
idata_final= np.array(ich)*np.array(a)          #奇数位数据与余弦函数相乘，得到一路的调制信号
qdata_final= np.array(qch)*np.array(b)          #偶数位数据与余弦函数相乘，得到另一路的调制信号

s=idata_final+qdata_final

plt.figure()
plt.subplot(2,1,1)
plt.plot(s)
plt.title('调制信号')
plt.axis([0,2400,-7,7])


def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise

s_noise=awgn(s,SNR)

plt.subplot(2,1,2)
plt.plot(s_noise)
plt.title("AWGN信号")
plt.axis([0,2400,-7,7])
plt.show()



data_frame=pd.DataFrame(data=s_noise,columns=["tiaozhi"])
print(data_frame)
data_frame.to_csv("data/16qam.csv",index=False)
