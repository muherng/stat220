#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 20:35:45 2017

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
        ll = ll + np.log(SPST.multinomial.pmf(eff[i,:],np.sum(eff[i,:]),theta[i,:]))
    return ll

#aprint(llikelihood(data,theta)) 

eff = eff_data(data)
q1_sum = eff.sum(axis=0)
like = SPST.multinomial.pmf(q1_sum,np.sum(q1_sum),theta)


#chinese restaurant process
def dir_process(pr,num_draws,alpha):
    theta = np.zeros(num_draws)
    buffet = []
    values = []
    theta = []
    for i in range(num_draws):
        if i == 0:
            draw = np.random.dirichlet(pr,1)
            values.append(draw)
            buffet.append(1)
        else:
            base = 1 - np.sum(buffet)/(i + alpha)
            roulette = [float(b)/(i + alpha) for b in buffet] + [base]
            roulette = [float(r)/np.sum(roulette) for r in roulette]
            #print(roulette)
            try:
                chosen = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
            except ValueError:
                print("ValueError")
            if chosen == len(roulette)-1:
                #print("base")
                draw = np.random.dirichlet(pr,1)
                values.append(draw)
                buffet.append(1)
            else:
                #print("cluster")
                draw = values[chosen]
                buffet[chosen] = buffet[chosen] + 1
        theta.append(draw)
        #print('theta')
        #print(theta)
    return theta
    
def bvi(theta):
    buffet = []
    values = []
    copy = np.copy(theta)
    while copy.size > 0:
        rem = []
        include = []
        for i in range(len(copy)):
            if np.array_equal(copy[i],copy[0]):
                rem.append(i)
            else:
                include.append(i)
        values.append(copy[0])
        buffet.append(len(rem))
        copy = copy[include,:]
    index = []
    for m in range(len(values)):
        v = values[m]
        sub = []
        for i in range(len(theta)):
            if np.array_equal(v,theta[i]):
                #print("APPEND SUB")
                sub.append(i)
        index.append(sub)
    return (buffet,values,index)

def llike(eff_row,val):
    n,k = data.shape
    ll = np.log(SPST.multinomial.pmf(eff_row,np.sum(eff_row),val))
    return ll
    
def gibbs(data,pr,alpha,iterations):
    theta = dir_process(pr,n,1)
    eff = eff_data(data)
    theta = np.array(theta)
    ll_list = [llikelihood(data,theta)]
    theta = theta.tolist()
    for it in range(iterations):
        print("iteration: " + str(it))
        for i in range(n):
            try:
                cut_theta = theta[:i] + theta[i+1:]
            except ValueError:
                print("ValueError")
            buffet,values,index = bvi(cut_theta)
            
            den = np.sum([np.log(y) for y in range(1,int(np.sum(eff[i,:])+4))])
            num = 0
            for j in range(eff.shape[1]):
                num += np.sum([np.log(y) for y in range(1,int(eff[i,j]+1))])
            fact = np.exp(num - den)
            discrete = []
            for j in range(len(values)):
                #print("complicated")
                #print(index[j])
                #print(eff[index[j],:])
                #print(np.exp(llike(eff[index[j],:],np.take(theta,index[j]))))
                
                theta = np.array(theta)
                discrete.append(np.exp(llike(eff[i,:],values[j]))*float(buffet[j])/n)
                theta = theta.tolist()
            normalize = float(6*alpha)/(alpha+n-1)*fact + np.sum(discrete)
            roulette = [float(d)/(normalize) for d in discrete] + [float(6*alpha)/(alpha+n-1)*fact/normalize]
            roulette = [float(r)/np.sum(roulette) for r in roulette]
            #print("roulette")
            #print(roulette) 
            chosen = int(np.where(np.random.multinomial(1,roulette) == 1)[0])
            if chosen == len(roulette) - 1:
                #print("base")
                theta[i] = np.random.dirichlet(np.add(pr,eff[i,:]),1)
            else:
                #print("cluster")
                theta[i] = values[chosen]
        #print(llikelihood(data,theta))
        #print("Theta")
        #print(theta)
        theta = np.array(theta)
        ll_list.append(llikelihood(data,theta))
        theta = theta.tolist()
    return (theta,ll_list)

print("GIBBS OUTPUT")
alpha = 1
iterations = 50
theta,ll_list = gibbs(data,pr,alpha,iterations)
plt.plot(ll_list)
plt.show()
#print(theta)