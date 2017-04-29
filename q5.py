#!/usr/bin/env python2
# -*- coding: utf-8 -*

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
                data[i,j] = 0
            if row[j] == 'c':
                data[i,j] = 1
            if row[j] == 'g':
                data[i,j] = 2
            if row[j] == 't':
                data[i,j] = 3
        i += 1
        
data = data[1:n+1,1:k+1]

pr = [1,1,1,1]
theta = np.random.dirichlet(pr,1)

def eff_data(data,lag):
    n,k = data.shape
    eff = np.zeros((4**lag,4))
    for i in range(n):
        print("bottleneck")
        for j in range(k-lag):
            ind = int(''.join(map(str, data[i,j:j+lag])).replace(".0",""),4)
            eff[ind,int(data[i,j+lag])] += 1
    return eff
                

def llikelihood(data,theta):
    n,k = data.shape
    eff = eff_data(data)
    ll = 0
    for i in range(n):
        ll = ll + np.log(SPST.multinomial.pmf(eff[i,:],np.sum(eff[i,:]),theta))
    return ll

#print(llikelihood(data,theta)) 
#
bottom = 13
trunc = bottom + 1
eff_all = []
for lag in range(bottom,trunc):
    eff_all.append(eff_data(data,lag))

log_list = []
for t in range(trunc-bottom):
    print("moving")
    eff = eff_all[t]
    tmp = 0
    for i in range(4**(t+bottom)):
        if i%1000 == 0:
            print(i)
        den = np.sum([np.log(y) for y in range(1,int(np.sum(eff[i,:])+4))])
        num = 0
        for j in range(4):
            num += np.sum([np.log(y) for y in range(1,int(eff[i,j]+1))])
        tmp += num - den
    log_list.append(4**(t+bottom)*np.log(6) + tmp)

log_bayes = [x1 - x2 for (x1,x2) in zip(log_list, [log_list[len(log_list)-1]]*len(log_list))]
print("Bayes Factor: ")
print(log_bayes)
        