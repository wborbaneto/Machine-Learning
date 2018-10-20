# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:26:43 2018

@author: AHCI
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:51:04 2018

@author: AHCI
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 17:46:35 2018

@author: AHCI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def sigmoid(z):
	g = 1.0/(1.0+np.exp(-z))
	return g


def sigmoid_grad(z):
	h = 1.0/(1.0+np.exp(-z))
	g = h*(1-h)
	return g



#def plot_class():
#    numC = 3
#slices = [c,n,a]
#yy = np.zeros([1,numC])
#for s in range(0,numC):
#    buffer = np.zeros([1, numC])
 #   buffer[:,s] = 1
 #   idx = np.all(y==buffer,1)
 #   plt.plot(x[idx,:],'.')
   
        
class CustomError(Exception):
    """Class to handle errors"""
    
    def __init__(self, value ='', message=''):
        """Get the value parsed as error.
        
         Parameters
        ----------
        value : undefined
            Origin of the error.
        """
        
        self.value = value
        self.message = message
        
        
    def __str__(self):
        """Value as a string."""
        
        return repr(str(self.value)+': '+self.message)    
     
class Algorithm():
    """Machine learning algoritm Class"""
    
    def __init__(self):
        """Not implemented"""
        
        pass
    
    def dataPartition(self, inputData, outputData, trainSize):
        """Partition the data into Train and Test arrays.
        
        Parameters
        ----------
        inputData : np.array([][], type = float64)
            A array cointaining the input data.
        outputData : np.aray([][], type = float64)
            A array containing the expected outputData corresponding to each
            input.
        trainSize : float64
            Decimal number between [0,1] related to the desired train/test 
            ratio.

        Returns
        -------
        trainData : float64
            Train data extracted from the (trainSize*100)% input's 
            first elements.
        testData : float64
            Test data extracted from the remaining (100 - trainSize*100)% 
            iputs's elements.
        expectedOutput : float64
            An array containing the expected classes for each input.
        """
        # Slicing the data in test and train
        trainData = inputData[0:int(trainSize*inputData.shape[0]),:]
        testData = inputData[int(trainSize*inputData.shape[0]):,:]
        trainOutput = outputData[0:int(trainSize*inputData.shape[0]),:]
        testOutput = outputData[int(trainSize*inputData.shape[0]):,:]
        return trainData,testData,trainOutput,testOutput
    
    
    def random_ini(self, inputData, outputData):
            """Randomly mixing the I/O pairs.
            
            Parameters
            ----------
            inputData : np.array(type = float64)
                A array cointaining the input data.
            outputData : np.aray(type = float64)
                A array containing the expected outputData corresponding to each
                input.
            
            Returns
            -------
            inputData : np.array(type = float64)
                A array cointaining the *mixed* input data.
            outputData : np.aray(type = float64)
                A array containing the expected outputData corresponding to each
                *mixed* input.
            """
            
            # Concatenating the columns in I/O to maintain data structure. 
            # In other words keep the I/O pairs attached.
            random_io = np.c_[inputData,outputData]
            np.random.shuffle(random_io)
            # Slicing the shuffled data to I/O format again.
            inputData = random_io[:,0:inputData.shape[1]]
            outputData = random_io[:,inputData.shape[1]:]
            return inputData,outputData
    
    
    def predict(self, pred, y):
        """Calculate the %correct classification ratio.
        
        Parameters
        ----------
        pred : np.array([][], type = float64)
            The predicted output's classes matching each input array.    
            For more information on the binary matrix format, please refer to
            Help folder
        y : float64
            An array containing the expected classes for each input
        
        Returns
        -------
        confArray = np.array([][], type = float64)
            The confusion matrix for the classification.
        """
        # Creating our confusing matrix nClass x nClass.
        confArray = np.zeros([y.shape[1],y.shape[1]])
        
        for [a,b] in np.column_stack((pred.argmax(1), 
                                      y.argmax(1))):
            confArray[a,b] += 1
            
        confArray *=100/np.sum(confArray,0)
        return confArray
        

class MultiLayerPerceptron(Algorithm):
    def __init__(self,inputLayer,hiddenLayer,outputLayer,
                 learningRate=1, regularizationParameter=0):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.lrp = learningRate
        self.rp = regularizationParameter
        self.thetaOne = np.zeros((self.hiddenLayer,self.inputLayer+1))
        self.thetaTwo = np.zeros((self.outputLayer,self.hiddenLayer+1))


    def initialize(self, inputData, outputData):
            self.x = inputData
            self.y = outputData
            return 1


    def train(self, trainSize, epochsNum, thetaOne=None, thetaTwo=None, randomize=1):
        #thetaOne = thetaOne if thetaOne is not None else self.thetaOne
        #thetaTwo = thetaTwo if thetaTwo is not None else self.thetaTwo
        
        if randomize:
            inputData, outputData = self.random_ini(self.x, self.y)
        # Slicing the data to the desired train size.
        self.trainInput,self.testInput,self.trainOutput,self.testOutput = self.dataPartition(inputData, outputData,trainSize)
        # To reduce the number of .T operations, do this beforehand.
        self.trainOutput,self.testOutput  = self.trainOutput.T,self.testOutput.T
        # Some self. definitions.
        [self.p,self.n] = self.trainInput.shape
        self.epochsNum = epochsNum
        self.JList = list()
        # The backpropagation Loop.
        for ep in range(0,epochsNum):
            a1, z2, a2, a3 = self.feedf(self.trainInput)
            self.JList += self.costf(self.trainOutput, a3)
            self.thetaOne, self.thetaTwo = self.backp(self.trainOutput, a3, a2, z2, a1)
        return self.thetaOne, self.thetaTwo
                   

    def backp(self, outputData, a3, a2, z2, a1, thetaOne=None, thetaTwo=None, p=None):
        thetaOne = thetaOne if thetaOne is not None else self.thetaOne
        thetaTwo = thetaTwo if thetaTwo is not None else self.thetaTwo
        p = p if p is not None else self.p
        
        delta3 = a3 - outputData
        thetaTwoBuffer = thetaTwo[:,1:]
        delta2 = (thetaTwoBuffer.T @ delta3)*sigmoid_grad(z2)
        
        thetaTwoGrad = (1/p) * (delta3 @ a2.T)
        thetaOneGrad = (1/p) * (delta2 @ a1.T)
        thetaTwoGrad[:,1:] += (self.rp/(p)) * thetaTwo[:,1:]
        thetaOneGrad[:,1:] += (self.rp/(p)) * thetaOne[:,1:]
        thetaOne = thetaOne - self.lrp*thetaOneGrad
        thetaTwo = thetaTwo - self.lrp*thetaTwoGrad
        return thetaOne, thetaTwo
    
    
    def costf(self, outputData, a3, thetaOne=None, thetaTwo=None, p=None):
        thetaOne = thetaOne if thetaOne is not None else self.thetaOne
        thetaTwo = thetaTwo if thetaTwo is not None else self.thetaTwo
        p = p if p is not None else self.p
        
        J = (1/p) * np.sum( np.sum( (-outputData) * np.log(a3)-
                                      (1 - outputData) * np.log(1 - a3))
                                )
        JReg = (self.rp/(2*p))*(np.sum(
                np.sum(thetaOne[:,1:]**2)) + 
                np.sum(np.sum(thetaTwo[:,1:]**2)))
        J += JReg      
        return [J]


    def feedf(self, inputData, thetaOne = None, thetaTwo = None, p=None):
        thetaOne = thetaOne if thetaOne is not None else self.thetaOne
        thetaTwo = thetaTwo if thetaTwo is not None else self.thetaTwo
        p = p if p is not None else self.p
        
        a1 = np.concatenate((np.ones((1,p)), inputData.T),0)
        z2 = thetaOne @ a1
        a2 = np.concatenate((np.ones((1,p)), sigmoid(z2)),0)
        z3 = thetaTwo @ a2
        a3 = sigmoid(z3)
        return a1, z2, a2, a3

    
    def test(self):
        [self.p,self.n] = self.testInput.shape
        _,_,_,a3 = self.feedf(self.testInput)
        cost = self.costf(self.testOutput, a3)
        confArray = self.predict(a3.T,self.testOutput.T)
        return cost,confArray
    
    
    def reset(self):
        self.__init__(self.inputLayer, self.hiddenLayer,
                      self.outputLayer)
        return True


    def costplot(self):
        JArray = np.array(self.JList)
        plt.plot(np.arange(len(self.JList)),self.JList)
        plt.axis([0, self.epochsNum, 0, JArray.max()])
        plt.ylabel('Cost function')
        plt.xlabel('No. of Epochs')
        plt.show()
        return True
    
    
class KNearestNeighbors(Algorithm):
    def __init__(self, inputData, outputData):
        """Initialize the datase that will be used in the algorithm.
        
        Parameters
        ----------
        inputData : np.array([][], type = float64)
            A array cointaining the input data for our algorithm. Each columm
            correspond a parameter and each row a input array.
        outputData : np.aray([][], type = float64)
            A array containing the expected outputData corresponding to each
            input. Each row correspond to the class which the input array 
            belongs. Must be in a winner-takes-all format.
        
        Returns
        -------
        None : nan
            Standard reponse.
        """
        
        self.inputData = inputData
        self.outputData = outputData
        
        
    def train(self, nNeighbors, trainSize, randomize = 1, dist = 'eDist'):
        """Link the k-nn algorithm methods.
        
        Parameters
        ----------
        nNeighbors : int
            Number of neighbors to be considered in the winner choose.
        trainSize : float64
            Decimal number between [0,1] related to the desired train/test 
            ratio.
        randomize : bool, default = 1
            Variable that express the need to randomize the I/O pairs sequence.
            By default the randomization will occur.
        
        Returns
        -------
        True: bool
            Filling empty space
        """
        
        self.nN = nNeighbors
        
        if randomize:
            self.inputData, self.outputData = self.random_ini(self.inputData, 
                                                              self.outputData)
            
        self.trainData,self.testData,_,self.y = self.dataPartition(self.inputData,
                                                                 self.outputData,
                                                                 trainSize)
        try:
            self.pred = self.nearestNeighbors(self.trainData, self.testData, 
                                              self.y, dist)
        except:
            raise 
            
        self.confArray = self.predict(self.pred,self.y)
        return None
      
    
    def nearestNeighbors(self, trainData, testData, y, dist):
        """Choose the Nearest Neighboor for each test input.
    
        Parameters
        ----------
         trainData : float64
            Train data extracted from the (trainSize*100)% input's 
            first elements.
        testData : float64
            Test data extracted from the remaining (100 - trainSize*100)% 
            iputs's elements.
        y : float64
            An array containing the expected classes for each input.
            
        Returns
        -------
        pred : np.array([][], type = float64)
            The predicted output's classes matching each input array.
        """
        
        listBuffer = list()
        # Distance calculation between each testData and trainData
        if dist == 'eDist':
            for arr in testData:
                listBuffer.append(self.euclideanDist(arr, trainData))
        elif dist == 'mDist':
            #raise CustomError(message = 'Not implemented yet.')
            for arr in testData:
                listBuffer.append(self.mahalanobisDist(arr, trainData))
        else: 
            raise CustomError(dist,' is not a distance definition.')
            
        # The distance organized in an array
        listBuffer = np.array(listBuffer).T
        
        # Finding the n-nearest values indexes
        idx = (listBuffer.T).argsort()[:,:self.nN]
        
        # y[idx] stores the classes to which each n-nearest value belongs
        #
        # The sum() is to find how many times each class has won, and
        # the argmax() finds which classes won most of the times
        winners = np.argmax(np.sum(self.outputData[idx],1),1)
        # Formatting the output prediction
        pred = np.zeros_like(y)
        pred[np.arange(y.shape[0]),winners] = 1
        return pred
    
    
    def euclideanDist(self, x1, x2):
        """Calculate the euclidean distance between x1 and x2
        
        Parameters
        ----------
        x1 : np.array([][], type = float64)
            First array. 
        x2 : np.array([][], type = float64)
            Second array.
            
        Returns
        -------
        eDist : np.array([][], type = float64)
            The calculated euclidean distance.
        """
        
        eDist = np.sqrt(np.sum((x2 - x1)**2,1))
        print(eDist.shape)
        return eDist
    
    
    def mahalanobisDist(self,x1, x2,S):
        """Calculate the mahalanobis distance between x1 and x2
        
        Parameters
        ----------
        x1 : np.array([][], type = float64)
            First array. 
        x2 : np.array([][], type = float64)
            Second array.
            
        Returns
        -------
        mDist : np.array([][], type = float64)
            The calculated mahalanobis distance.
        """
        A = x1[None].T - x2.T
        mDist = np.diag(np.sqrt(abs(A.T @ S @ A)))
        return mDist
    
    
class kmeans(KNearestNeighbors):
    
    def train(self, trainSize, centroidNum, randomize = 1, dist = 'eDist'):
        self.centroidNum = centroidNum
        self.centroid = np.random.rand(self.centroidNum,self.inputData.shape[1])
        
        if randomize:
            self.inputData, self.outputData = self.random_ini(self.inputData, 
                                                              self.outputData)
        self.trainData,self.testData,_,self.y = self.dataPartition(self.inputData,
                                                                 self.outputData,
                                                                 trainSize)
        self.nN = 1
        error = 1
        intNum = 0
        S = np.linalg.inv(np.cov(self.trainData.T))
        while np.any(error > 0.0001) or intNum > 10000:
            self.winners = self.cluster_assigment(self.trainData,self.centroid,dist,S)
            newCentroid = self.move_centroid(self.trainData, self.winners)
            error = abs(newCentroid - self.centroid)
            self.centroid = newCentroid
            intNum += 1
        return self.centroid,error
    
    def cluster_assigment(self, trainData, testData, dist,S):
        listBuffer = list()
        #trainData = x
        #testData = centroid
        # Distance calculation between each testData and trainData
        if dist == 'eDist':
            for arr in testData:
                listBuffer.append(self.euclideanDist(arr, trainData))
        elif dist == 'mDist':
            for arr in testData:
                listBuffer.append(self.mahalanobisDist(arr,trainData,S))
        else: 
            raise CustomError(dist,' is not a distance definition.')
            
        # The distance organized in an array
        listBuffer = np.array(listBuffer).T
        # Finding the n-nearest values indexes
        winners = (listBuffer).argsort()[:,:self.nN]
        return winners
    
    
    def move_centroid(self, trainData, winners):
        newCentroid = list()
        for c in range(0,self.centroidNum):
            a = (winners==c).ravel()
            if np.any(a) == False:
                return np.random.rand(self.centroidNum,self.inputData.shape[1])
            newCentroid.append(np.mean(trainData[a],0))
            
        newCentroid = np.array(newCentroid)
        return newCentroid

    def test(self):
        centroid = self.centroid
        outputData = self.y
        inputData = self.testData
        winner = self.cluster_assigment(inputData,centroid,'eDist')
        pred = np.zeros_like(outputData)
        pred[np.arange(outputData.shape[0]),winner.ravel()] = 1
        confArray = self.predict(pred,outputData)
        return confArray
    
    
if __name__ == "__main__":
    irisdf = pd.read_csv('data/iris.csv')
    iris = np.array(irisdf.values)

    x = np.array(iris[:,0:4],dtype='float64')
    y = np.concatenate(
            (np.tile(np.array([1,0,0]),(50,1)),np.tile(np.array([0,1,0]),(50,1)),
             np.tile(np.array([0,0,1]),(50,1))),
             0)


        
        