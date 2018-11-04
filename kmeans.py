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
classes = list(['Class1','Class2','Class3']) 

irisdf = pd.read_csv('data/iris.csv')
iris = np.array(irisdf.values[:,0:4],dtype='float64')

a,b,c = 50,50,50
cl1 = iris[0:a,:]
cl2 = iris[a:a+b,:]
cl3 = iris[a+b:a+b+c,:]

cl3 = np.array(cl3.values)
cl2 = np.array(cl2.values)
cl1 = np.array(cl1.values)
x = np.r_[cl1,cl2,cl3]
y = conv.out_b([a,b,c])
x,y = alg.random_ini(x,y)

x =  (x - np.min(x,0))/(np.max(x,0) - np.min(x,0))

dist = 'mDist'

km = alg.kmeans(x,y)  
centroid,winners = km.train(1,3,randomize=0,dist=dist)


xdf = pd.DataFrame(x, columns=labels)
xdf['Class'] = conv.dec2str(winners,classes)

mks = ['o','^','s','X']
pll = ['c','m','0.6','b']

def cluster_matrix(xdf,centroid,labels,classes,mks = None,pll=None):
    numC = len(classes)
    mks = mks if mks is not None else ["o" for n in range(numC)] + ["X"]
    pll = pll if pll is not None else "deep"
    
    classes.append('Centroid')
    
    centroidf = pd.DataFrame(centroid, columns = labels )
    centroidf['Class'] = ['Centroid' for n in range(numC)]
    xdf = pd.concat([xdf,centroidf], ignore_index=True)
    
    g = sns.PairGrid(xdf, hue="Class",
                 hue_order=classes,
                 palette= pll,
                 hue_kws={"s":[30 for n in range(numC)]+[100],
                          "marker": mks})
    g.map_offdiag(plt.scatter, linewidth=0.02, edgecolor="k")
    g.map_diag(plt.hist)
    g.add_legend()
    return g
    
g1 = cluster_matrix(xdf, centroid ,labels,classes, mks, pll) 
    
g1.fig.suptitle('K-means ('+dist+') Output')


