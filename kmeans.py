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

#labels = list(['RPM','Speed','Throttle','Engine Load'])
labels = list(['sepalLength','sepalWidth','petalLenght','petalWidth'])
cl3 = pd.read_csv('data/agressiva.csv')
cl2 = pd.read_csv('data/normal.csv')
cl1 = pd.read_csv('data/lenta.csv')

irisdf = pd.read_csv('data/iris.csv')
iris = np.array(irisdf.values[:,0:4],dtype='float64')

#a,b,c = cl1.shape[0],cl2.shape[0],cl3.shape[0]
#r = [a,b,c]
#r = np.min(r)

a,b,c = 50,50,50
cl1 = iris[0:a,:]
cl2 = iris[a:a+b,:]
cl3 = iris[a+b:a+b+c,:]

#cl3 = np.array(cl3.values)
#cl2 = np.array(cl2.values)
#cl1 = np.array(cl1.values)

x = np.r_[cl1,cl2,cl3]
y = conv.out_b([a,b,c])

x =  (x - np.min(x,0))/(np.max(x,0) - np.min(x,0))

dist = 'eDist'
    
km = alg.kmeans(x,y)  
centroid,error = km.train(0.8,3,randomize=1,dist=dist)
confArray = km.test()

print(centroid)
p1 = 2
p2 = 3

#plt.figure(1)
#plt.plot(x[0:a,p1],x[0:a,p2],'go',markersize=4)
#plt.plot(x[a:a+b,p1],x[a:a+b,p2],'y^',markersize=4)
#plt.plot(x[a+b:a+b+c,p1],x[a+b:a+b+c,p2],'rs',markersize=4)
#plt.plot(centroid[:,p1],centroid[:,p2],'bX',markersize=10)
#plt.xlabel(labels[p1]),plt.ylabel(labels[p2])
#plt.legend(['Setosa','Versicolor','Virginica','Centroids'])
#if dist == 'eDist': plt.title('K-means (eDist) Input')
#else: plt.title('K-means (mDist) Input')

#x = km.trainData
#dep = km.winners
#plt.figure(2)    
#xcl1,xcl2,xcl3 = x[(km.winners==0).ravel()],x[(km.winners==1).ravel()],x[(km.winners==2).ravel()]
#plt.plot(xcl1[:,p1],xcl1[:,p2],'go',markersize=4)
#plt.plot(xcl2[:,p1],xcl2[:,p2],'y^',markersize=4)
#plt.plot(xcl3[:,p1],xcl3[:,p2],'rs',markersize=4)
#plt.plot(centroid[:,p1],centroid[:,p2],'bX',markersize=10)
#plt.xlabel(labels[p1]),plt.ylabel(labels[p2])
#plt.legend(['Setosa','Versicolor','Virginica','Centroids'])
#plt.xlim([0,1])
#plt.ylim([0,1])
#if dist == 'eDist': plt.title('K-means (eDist) Output')
#else: plt.title('K-means (mDist) Output')

x = km.testIn
xdf = pd.DataFrame(x, columns=labels)
xdf['Class'] = conv.dec2str(km.testOut,['Class1','Class2','Class3'])
#g1=sns.pairplot(xdf,hue='Class',markers=['o', '^', 's'], palette = ['c','m','k'], plot_kws=dict(edgecolor='k',linewidth=0,s=30))
#g2=sns.pairplot(xdf,kind='reg',hue='Class',markers=['o', '^', 's'], palette = {"Calm":"g","Normal":"y","Aggressive":"r"}, 
#             plot_kws={'scatter_kws':{'linewidth':0,'s':30}})

#xdf['Class'] = conv.bin2str(km.eOut,['Calm','Normal','Agressive'])
#g3=sns.pairplot(xdf,hue='Class',markers=['o', '^', 's'], palette = {"Calm":"g","Normal":"y","Agressive":"r"}, plot_kws=dict(edgecolor='k',linewidth=0,s=30))
#km.testIn
#g4=sns.pairplot(xdf,kind='reg',hue='Class',markers=['o', '^', 's'], palette = {"Calm":"g","Normal":"y","Aggressive":"r"}, 
#             plot_kws={'scatter_kws':{'linewidth':0,'s':30}})

#g1.fig.suptitle('K-means ('+dist+') Output')
#g2.fig.suptitle('K-means ('+dist+') Regressed Output')
#g3.fig.suptitle('K-means ('+dist+') Input')
#g4.fig.suptitle('K-means ('+dist+') Regressed Input')

g5 = sns.PairGrid(xdf,hue='Class')
g5 = g5.map_offdiag(plt.scatter)
print(confArray)
'''
caMax,caMin,caMean,caStd = list(),list(),list(),list()

trainArr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for trainSize in trainArr:
    confArrayList = list()
    for n in range(0,30):
       km.train(trainSize,3,dist=dist)
       confArray = km.test()
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
'''

