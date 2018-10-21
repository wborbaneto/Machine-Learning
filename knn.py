# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 20:42:09 2018

@author: wborbaneto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import algorithm as alg
import pyconv as conv

irisdf = pd.read_csv('data/iris.csv')
iris = np.array(irisdf.values)

x = np.array(iris[:,0:4],dtype='float64')
y = conv.out_b([50,50,50])
x = conv.normalize(x)

caMax,caMin,caMean,caStd = list(),list(),list(),list()

x =  (x - np.min(x,0))/(np.max(x,0) - np.min(x,0))
met = alg.KNearestNeighbors(x,y)
trainArr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for trainSize in trainArr:
    confArrayList = list()
    for n in range(0,100):
       met.train(7, trainSize,dist = 'mDist')
       confArrayList.append(met.confArray)
       
    cal = np.array(confArrayList) 
    caMax.append(np.diag(np.max(cal,0)))
    caMin.append(np.diag(np.min(cal,0)))
    caMean.append(np.diag(np.mean(cal,0)))
    caStd.append(np.diag(np.std(cal,0)))
    
caMax = np.array(caMax)
caMin = np.array(caMin)
caMean = np.array(caMean)
caStd = np.array(caStd)
cM = np.mean(cal,0)

print(np.max(caMax))
print(np.min(caMin))
print(np.mean(caStd))

plt.figure(1)
plt.plot(trainArr,caMax[:,:])
plt.xlabel('Train dataset size')
plt.ylabel('Max value of successful classification')
plt.title("Maximum Correct Classification related to Train size (K = 7)")
plt.legend(['Setosa','Versicolor','Virginica'])
plt.grid()
plt.figure(2)
plt.plot(trainArr,caMin[:,:])
plt.xlabel('Train dataset size')
plt.ylabel('Min value of successful classification')
plt.title("Minimum Correct Classification related to Train size (K = 7)")
plt.legend(['Setosa','Versicolor','Virginica'])
plt.grid()
plt.figure(3)
plt.plot(trainArr,caMean[:,:])
plt.xlabel('Train dataset size')
plt.ylabel('Mean value of successful calssification')
plt.title("Mean of Correct Classification related to Train size (K = 7)")
plt.legend(['Setosa','Versicolor','Virginica'])
plt.grid()
plt.figure(4)
plt.plot(trainArr,caStd[:,:])
plt.xlabel('Train dataset size')
plt.ylabel('Std value of successful classification')
plt.title("Deviation of Correct Classification related to Train size (K = 7)")
plt.legend(['Setosa','Versicolor','Virginica'])
plt.grid()