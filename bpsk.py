# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 10:12:55 2020

@author: 33578
"""

from scipy import signal, special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False ####为了让title正常显示中文

T = 1               #基带信号宽度，也就是频率
nb = 200            #定义传输的比特数
delta_T = T/200     #采样间隔
fs = 1/delta_T      #采样频率
fc = 2/T           #载波频率
SNR = 20            #信噪比

t = np.arange(0, nb*T, delta_T)
N = len(t)

# 产生基带信号
data = [1 if x > 0.5 else 0 for x in np.random.randn(1, nb*20)[0]]  #调用随机函数产生任意在0到1的1*nb的矩阵，大于0.5显示为1，小于0.5显示为0
print("data:",data)
data0 = []                             #创建一个1*nb/delta_T的零矩阵
for q in range(nb):
    data0 += [data[q]]*int(1/delta_T)  #将基带信号变换成对应波形信号
# 调制信号的产生
#plt.plot(data0)
data1 = []      #创建一个1*nb/delta_T的零矩阵
datanrz = np.array(data)*2-1              #将基带信号转换成极性码,映射
for q in range(nb):
    data1 += [datanrz[q]]*int(1/delta_T)  #将极性码变成对应的波形信号
plt.figure()
plt.title("极性码")
plt.plot(data1)
plt.show()
a = []     #余弦函数载波
for j in range(int(N)):
    a.append(np.math.sqrt(2/T)*np.math.sin(2*np.math.pi*fc*t[j]))    #余弦函数载波
plt.figure()
plt.title("载波")
plt.plot(a)
plt.axis([500,1500,-2,2])
s=np.array(data1)*np.array(a)
plt.figure()
plt.subplot(2,1,1)
plt.plot(s)
plt.title("调制信号")

plt.show()


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
plt.show()

data_frame=pd.DataFrame(data=s_noise,columns=["tiaozhi"])
print(data_frame)
data_frame.to_csv("data/bpsk.csv",index=False)



