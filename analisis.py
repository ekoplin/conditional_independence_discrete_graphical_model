#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:50:40 2024

@author: eric
"""
import pandas as pd
import numpy as np 
from discrete_gm_nonpos import discrete_graphical_model

WS = pd.read_excel('ecu_ecv13-14.xlsx')


X = WS[['d_cm','d_nutr','d_satt','d_educ','d_elct','d_wtr','d_sani','d_hsg','d_ckfl','d_asst']].to_numpy()
Y = pd.Categorical(WS.area).codes.reshape(n,1)
Y = pd.Categorical(WS.region).codes.reshape(n,1)


n=X.shape[0]
indx_nan=np.isnan(X).any(1)|np.isnan(Y).any(1)
X = X[~indx_nan,:]
Y = Y[~indx_nan,:]
ci=discrete_graphical_model(c=0).estimate_CI(X>0, Y>0)

ci=discrete_graphical_model(c=.1).estimate_CI(X>0, np.zeros((n,0))>0)
