#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:02:21 2017

@author: morrisyau
"""

import csv
import numpy as np
import numpy.random as npr 
import scipy.special as sps
import scipy.stats as SPST
import pdb 
from fractions import gcd
import matplotlib.pyplot as plt


n = 18
k = 105
data = np.zeros((n+1,k+1))

with open('SequenceData.csv','rb') as f:
    reader = csv.reader(f)
    i = 0
    for row in reader:
        for j in range(1,k+1):
            if row[j] == 'a':
                data[i,j] = 1
            if row[j] == 'c':
                data[i,j] = 2
            if row[j] == 'g':
                data[i,j] = 3
            if row[j] == 't':
                data[i,j] = 4
        i += 1
        
data = data[1:n+1,1:k+1]

pr = [1,1,1,1]
theta = np.random.dirichlet(pr,1)

def eff_data(data):
    n,k = data.shape
    eff = np.zeros((n,4))
    for i in range(n):
        for j in range(k):
            eff[i,int(data[i,j]-1)] =  eff[i,int(data[i,j]-1)] + 1
    return eff
                

def llikelihood(data,theta):
    n,k = data.shape
    eff = eff_data(data)
    ll = 0
    for i in range(n):
        ll = ll + np.log(SPST.multinomial.pmf(eff[i,:],np.sum(eff[i,:]),theta))
    return ll

print(llikelihood(data,theta)) 

eff = eff_data(data)
q1_sum = eff.sum(axis=0)
like = SPST.multinomial.pmf(q1_sum,np.sum(q1_sum),theta)
print(like)

den = np.sum([np.log(y) for y in range(1,int(np.sum(q1_sum)+4))])
num = 0
for j in range(4):
    num += np.sum([np.log(y) for y in range(1,int(q1_sum[j]+1))])
log1 = np.log(6) + num-den
#fact1 = 6*np.exp(num - den)

tmp = 0
for i in range(n):
    den = np.sum([np.log(y) for y in range(1,int(np.sum(eff[i,:])+4))])
    num = 0
    for j in range(4):
        num += np.sum([np.log(y) for y in range(1,int(eff[i,j]+1))])
    tmp += num - den
log2 = n*np.log(6) + tmp
#fact2 = 6**n*np.exp(tmp)

log_bayes = log1 - log2
print("Question 2 Answer: " + str(np.exp(log_bayes)))
#print(np.exp(log_bayes))    
    
  

        