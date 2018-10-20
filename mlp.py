# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:51:04 2018

@author: wborbaneto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import algorithm as alg
import pyconv as conv

def s_anal(cal):
    caMax = np.max(cal,0)
    caMin =np.min(cal,0)
    caMean = np.mean(cal,0)
    caStd = np.std(cal,0)
    return caMax,caMin,caMean,caStd
    

irisdf = pd.read_csv('data/iris.csv')
iris = np.array(irisdf.values)

x = np.array(iris[:,0:4],dtype='float64')
y = conv.out_b([50,50,50])
x = conv.normalize(x)

inputLayer   = 4;
hiddenLayer  = 4;
outputLayer  = 3;
learningRate = 0.3;
regularizationParameter = 0;

net = alg.MultiLayerPerceptron(inputLayer, hiddenLayer, outputLayer,
                               learningRate, regularizationParameter)

net.initialize(x,y)


caMax,caMin,caMean,caStd = list(),list(),list(),list()

trainArr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for trainSize in trainArr:
    confArrayList = list()
    for n in range(0,30):
       net.train(trainSize,100)
       cost, confArray = net.test()
       confArrayList.append(confArray)
       
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


        