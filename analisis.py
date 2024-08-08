#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:50:40 2024

@author: eric
"""
# import pandas as pd
import os
import numpy as np 
from discrete_gm_nonpos import discrete_graphical_model

# WS = pd.read_excel('ecu_ecv13-14.xlsx')
# X = WS[['d_cm','d_nutr','d_satt','d_educ','d_elct','d_wtr','d_sani','d_hsg','d_ckfl','d_asst']].to_numpy()

# Ynone = np.zeros((X.shape[0],0))
# Yarea = pd.Categorical(WS.area).codes.reshape(X.shape[0],1)
# Yregion = pd.get_dummies(pd.Categorical(WS.region)).to_numpy()# convert to binary multivariate

# data = {'X':X,'Ynone':Ynone,'Yarea':Yarea,'Yregion':Yregion}
# np.savez('ecu_ecv13-14.npz',**data)
data=np.load('ecu_ecv13-14.npz')


#for (Y,name) in [(Ynone,"none"), (Yarea,"area"),(Yregion,"region")]:
for name in ["Ynone", "Yarea","Yregion"]:
    Y=data[name]
    X=data['X']
    indx_nan=np.isnan(X).any(1)|np.isnan(Y).any(1)
    Xclean = X[~indx_nan,:]
    Yclean = Y[~indx_nan,:]
    ci=discrete_graphical_model(np.linspace(1, 10,10)).estimate_CI(Xclean>0, Yclean>0)# only binary data allowed
    os.mkdir("./ecu_ecv13-14_covar/")
    for ic in range(len(ci['conserv'])):
        np.savetxt("./ecu_ecv13-14_covar/"+name+"_conservative_c"+str(ic)+".txt", ci['conserv'][ic] , fmt="%5i")
        np.savetxt("./ecu_ecv13-14_covar/"+name+"_nconservative_c"+str(ic)+".txt", ci['nconserv'][ic] , fmt="%5i")

