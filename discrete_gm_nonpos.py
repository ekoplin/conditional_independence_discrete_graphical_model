#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 19:33:07 2024

@author: eric
"""

import numpy as np
from itertools import combinations
def int2bin(x, bits):
    return np.array([int(i) for i in bin(x)[2:].zfill(bits)])
# n=100
# p=4
# q=3
# X = np.random.binomial(1, .1, (n,p))
# Y = np.random.binomial(3, .1, (n,q))

class discrete_graphical_model:
    def __init__(self,c=.1,conservative=True):
        self.c = c
        self.conservative = conservative
    def _bic(self,X,indx_v,indx_w):
        #X[:,list(set(range(X.shape[1])).difference(indx_v+indx_w))]=0# do not accout for the variables not appearing in the index
        Xnz = np.zeros_like(X)
        Xnz[:,indx_v+indx_w]=X[:,indx_v+indx_w]
        n,p = X.shape
        # find all configurations presented in the data (applied to subvector)        
        Xint = X.dot(np.power(2,np.arange(p-1,0-1,-1)))
        keys = np.unique(Xint)
        N_av_aw = dict()
        for key in keys:
            N_av_aw[key] = sum(Xint==key)
        N_aw = dict()
        #P_av_given_aw = dict()
        logP_av_given_aw = dict()
        for key1 in keys:
            if len(indx_w)>0:
                N_aw[key1]=N_av_aw[key1]
                for key2 in keys:
                   if (key1!=key2) and all(int2bin(key1, p)[indx_w]==int2bin(key2, p)[indx_w]):
                       N_aw[key1]+=N_av_aw[key2]
            else:
                N_aw[key1]=n
            #P_av_given_aw[key1]=N_av_aw[key1]/N_aw[key1]
            logP_av_given_aw[key1]=np.log(N_av_aw[key1])-np.log(N_aw[key1])
            
        # check <=1
        #P_av_given_aw
        
        lpl = 0
        for key in keys:
            #lpl += np.log(P_av_given_aw[key])*N_av_aw[key]
            lpl += logP_av_given_aw[key]*N_av_aw[key]
        
        #bic = lpl - self.c*pow(len(indx_v)+len(indx_w),len(indx_w))*np.log(n)
        bic = lpl - self.c*pow(len(np.unique(X)),len(indx_w))*np.log(n)
        return(bic)
    def estimate_CI(self,X,Y):
        # estimate the neighbourhood of every index
        # X predictors
        # Y covariates
        # c positive constant (regularization)
        YX = np.hstack((Y,X))
        q = Y.shape[1]# covariates
        p = X.shape[1]
        
        NE = np.zeros((p,p),dtype=bool)
        indx_all = set(range(p)) 
        for i in range(p):
            indx_v = [i]
            indx_no_v =indx_all.difference(indx_v)
            
            bic_v = []
            ne_v  = []
            for ne in range(len(indx_all)):
                for indx_w in combinations(indx_no_v, ne):
                    #print(indx_v,indx_w)
                    ne_v.append(indx_w)
                    bic_v.append(self._bic(YX,[q+i for i in indx_v],list(range(q))+[q+j for j in indx_w]))
            ne_v_optim = ne_v[max(enumerate(bic_v), key=lambda x: x[1])[0]]
            print(bic_v)
            print(max(enumerate(bic_v), key=lambda x: x[1])[0])
            
            NE[i,ne_v_optim]=True
        
        # simetrization
        if self.conservative:
            NE = NE & np.transpose(NE)
        else:
            NE = NE | np.transpose(NE)
        return(NE)
