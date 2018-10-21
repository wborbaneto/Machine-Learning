# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 21:16:57 2018

@author: wborbaneto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import algorithm as alg
import pyconv as conv

labels = list(['sepalLength','sepalWidth','petalLenght','petalWidth'])

irisdf = pd.read_csv('data/iris.csv')
iris = np.array(irisdf.values[:,0:4],dtype='float64')

a,b,c = 50,50,50
cl1 = iris[0:a,:]
cl2 = iris[a:a+b,:]
cl3 = iris[a+b:a+b+c,:]

x = np.r_[cl1,cl2,cl3]
y = conv.out_b([a,b,c])

x =  (x - np.min(x,0))/(np.max(x,0) - np.min(x,0))

dist = 'mDist'
    
km = alg.kmeans(x,y)  
centroid,error = km.train(0.5 ,3,randomize=1,dist=dist)
confArray = km.test()
print(confArray)

