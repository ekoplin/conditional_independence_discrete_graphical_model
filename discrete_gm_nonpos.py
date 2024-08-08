#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:33:07 2024

@author: eric
"""

import numpy as np
from itertools import combinations
import multiprocessing
from functools import partial
def int2bin(x, bits):
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])

class discrete_graphical_model:
    def __init__(self,c=np.linspace(.1,1,10),ncores=None):
        self.c = c.reshape(-1,1)# column
        self.ncores = ncores
    def _bic(self,X,indx_v,indx_w,ne_size):
        Xnz = np.zeros_like(X)
        Xnz[:,indx_v+indx_w]=X[:,indx_v+indx_w]
        n,p = X.shape
        # find all configurations presented in the data (applied to subvector)        
        Xint = Xnz.dot(np.power(2,np.arange(p-1,0-1,-1)))
        keys = np.unique(Xint)
        N_av_aw = dict()
        for key in keys:
            N_av_aw[key] = sum(Xint==key)
        N_aw = dict()
        logP_av_given_aw = dict()
        for key1 in keys:
            if len(indx_w)>0:
                N_aw[key1]=N_av_aw[key1]
                for key2 in keys:
                   if (key1!=key2) and all(int2bin(key1, p)[indx_w]==int2bin(key2, p)[indx_w]):
                       N_aw[key1]+=N_av_aw[key2]
            else:
                N_aw[key1]=n
            logP_av_given_aw[key1]=np.log(N_av_aw[key1])-np.log(N_aw[key1])
        
        lpl = 0
        for key in keys:
            lpl += logP_av_given_aw[key]*N_av_aw[key]
        
        #bic = lpl - self.c*pow(len(np.unique(X)),len(indx_w))*np.log(n)
        bic = lpl - self.c*pow(len(np.unique(X)),ne_size)*np.log(n)
        return(bic)
    def compute_ne_i(self,i,X,Y):
        YX = np.hstack((Y,X))
        q = Y.shape[1]# covariates
        p = X.shape[1]
        
        indx_v = [i]
        indx_all = set(range(p)) 
        indx_no_v =indx_all.difference(indx_v)
        
        bic_v = []
        ne_v  = []
        for ne in range(len(indx_all)):
            for indx_w in combinations(indx_no_v, ne):
                ne_v.append(indx_w)
                bic_v.append(self._bic(YX,[q+i for i in indx_v],list(range(q))+[q+j for j in indx_w],len(indx_w)))
        #ne_v_optim = ne_v[max(enumerate(bic_v), key=lambda x: x[1])[0]]
        #NE[i,ne_v_optim]=True
        ne_v_optim_indx=np.argmax(np.hstack(bic_v),axis=1,keepdims=True)
        ne_v_optim     = np.zeros((len(self.c),p),dtype=bool)
        for ic in range(len(self.c)):
            ne_v_optim[ic,ne_v[int(ne_v_optim_indx[ic])]]=True 
        #NElst.append(ne_v_optim)
        return(ne_v_optim)
    def estimate_CI(self,X,Y):
        # estimate the neighbourhood of every index
        # X predictors
        # Y covariates
        # c positive constant (regularization)
        
        
        #NElst = list()
        
        
        with multiprocessing.Pool(self.ncores) as pool:
            NElst=pool.map(partial(self.compute_ne_i,X=X,Y=Y), range(X.shape[1]))
        
            
        NE=np.stack(NElst,2)# |c|xpxp <=> c,neighbor, variable
        # simetrization
        NE_conserv  = NE & np.moveaxis(NE,-1,-2)
        NE_nconserv = NE | np.moveaxis(NE,-1,-2)
        return({'conserv' : np.split(NE_conserv, 1, 0)[0], 'nconserv' : np.split(NE_nconserv, 1, 0)[0]})
        # if self.conservative:
        #     NE = NE & np.transpose(NE)
        # else:
        #     NE = NE | np.transpose(NE)
        # return(NE)
