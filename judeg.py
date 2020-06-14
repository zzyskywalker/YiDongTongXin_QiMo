# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 01:55:15 2020

@author: 33578
"""

import pandas as pd
import os
from collections import Counter
import numpy as np
import joblib


clf=joblib.load(model)

bpsk_test=pd.read_csv