# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:46:22 2018

@author: wborbaneto
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import algorithm as alg
import pyconv as conv

x = km.testIn
xdf = pd.DataFrame(x, columns=labels)
xdf['Class'] = conv.dec2str(km.testOut,['Class1','Class2','Class3'])
centroidf = pd.DataFrame(centroid, columns =labels )
centroidf['Class'] = ['Centroid','Centroid','Centroid']
xdf1 = pd.concat([xdf,centroidf], ignore_index=True)

g1 = sns.PairGrid(xdf1, hue="Class",
                 hue_order=["Class1", "Class2", "Class3", "Centroid"],
                 palette=["c","m","0.5", "b"],
                 hue_kws={"s": [30, 30, 30, 100],
                          "marker": ["o","^","s","X"]})
g1.map_offdiag(plt.scatter, linewidth=0.02, edgecolor="k")
g1.map_diag(plt.hist)
g1.add_legend()
g1.fig.suptitle('K-means ('+dist+') Output')

xdf['Class'] = conv.bin2str(km.eOut,['Calm','Normal','Agressive'])
xdf1 = pd.concat([xdf,centroidf], ignore_index=True)

g2 = sns.PairGrid(xdf1, hue="Class",
                 hue_order=["Calm", "Normal", "Agressive", "Centroid"],
                 palette={"Calm":"g","Normal":"y","Agressive":"r","Centroid":"b"},
                 hue_kws={"s": [30, 30, 30, 100],
                          "marker": ["o","^","s","X"]})
g2.map_offdiag(plt.scatter, linewidth=0.02, edgecolor="k")
g2.map_diag(plt.hist)
g2.add_legend()
g2.fig.suptitle('K-means ('+dist+') Input')

caMax,caMin,caMean,caStd = list(),list(),list(),list()

trainArr = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

for trainSize in trainArr:
    confArrayList = list()
    for n in range(0,300):
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

plt.figure(3)
plt.plot(trainArr,caMax[:,:])
plt.xlabel('Train dataset size')
plt.ylabel('Max value of successful classification')
plt.title("Maximum Correct Classification related to Train size")
plt.legend(['Setosa','Versicolor','Virginica'])
plt.grid()
plt.figure(4)
plt.plot(trainArr,caMin[:,:])
plt.xlabel('Train dataset size')
plt.ylabel('Min value of successful classification')
plt.title("Minimum Correct Classification related to Train size")
plt.legend(['Setosa','Versicolor','Virginica'])
plt.grid()
plt.figure(5)
plt.plot(trainArr,caMean[:,:])
plt.xlabel('Train dataset size')
plt.ylabel('Mean value of successful calssification')
plt.title("Mean of Correct Classification related to Train size")
plt.legend(['Setosa','Versicolor','Virginica'])
plt.grid()
plt.figure(6)
plt.plot(trainArr,caStd[:,:])
plt.xlabel('Train dataset size')
plt.ylabel('Std value of successful classification')
plt.title("Deviation of Correct Classification related to Train size")
plt.legend(['Setosa','Versicolor','Virginica'])
plt.grid()