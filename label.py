# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:30:19 2020

@author: 33578
"""

import pandas as pd

data_bpsk=pd.read_csv("feature/bpsk.csv")
data_qpsk=pd.read_csv("feature/qpsk.csv")
data_16qam=pd.read_csv("feature/16qam.csv")

data_bpsk["label"]=0
data_qpsk["label"]=1
data_16qam["label"]=2

data_labeld=pd.concat([data_bpsk,data_qpsk,data_16qam],ignore_index=True)

data_bpsk.to_csv("label/bpsk.csv",index=False)
data_qpsk.to_csv("label/qpsk.csv",index=False)
data_16qam.to_csv("label/16qam.csv",index=False)

data_labeld.to_csv("label/train.csv",index=False)