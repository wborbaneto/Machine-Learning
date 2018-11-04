# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 17:26:43 2018

@author: wborbaneto
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

def random_ini(inputData, outputData):
            # Concatenating the columns in I/O to maintain data structure. 
            # In other words keep the I/O pairs attached.
            random_io = np.c_[inputData,outputData]
            np.random.shuffle(random_io)
            # Slicing the shuffled data to I/O format again.
            inputData = random_io[:,0:inputData.shape[1]]
            outputData = random_io[:,inputData.shape[1]:]
            return inputData,outputData
        
class CustomError(Exception):
    """Class to handle errors."""
    
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
    """Machine learning algoritm Class.
    
    This Class holds data manipulation and evaluation methods.
    """
    
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
    """Class that represents a MLP"""
    
    def __init__(self,inputLayer,hiddenLayer,outputLayer,
                 learningRate=1, regularizationParameter=0):
        """Initialize the MLP architeture.
        
        Parameters
        ----------
        inputLayer : int32
            Number of neurons on the input layer.
        hiddenLayer : int32
            Number of neurons on the hidden layer.
        outputLayer : int32
            Number of neurons on the output layer.
        learningRate : float64, default = 1
            Learning rate parameter for our MLP. 
            Utilized on the Gradient Descent
        regularizationParameter : float64, default = 0
            May be added if you want to regularize the Cost function.
            
        """
        # Defining the standadrd states of our MLP.
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
        self.outputLayer = outputLayer
        self.lrp = learningRate
        self.rp = regularizationParameter
        # Theta are the weights of our MLP. They follow the general rule
        # ThetaL.shape() = [len(L+1), len(L)+1], where L is a Layer
        self.thetaOne = np.zeros((self.hiddenLayer,self.inputLayer+1))
        self.thetaTwo = np.zeros((self.outputLayer,self.hiddenLayer+1))
        return None

    def initialize(self, inputData, expectedOutput):
            """Initialize the inputs.
            
            This is done so there's no need to reinitialize the object itself
            if the user wants to change the input.
            
            Parameters
            ----------
            inputData : np.array([][], type = float64)
                Data that will be classified.
            outputData :np.array([][], type = float64)
                Expected output (Supervisioned algorithm).
             
            """
            self.x = inputData
            self.y = expectedOutput
            return None


    def train(self, trainSize, epochsNum, thetaOne=None, thetaTwo=None, randomize=1):
        """Training our MLP.
        
        Parameters
        ----------
        trainSize : float64
            The factor that corresponds to what % of the input data corresponds
            to the train part. 
            ex. trainData = inputData[0:trainSize*inputData.shape(0)]
        epochsNum : int32
            Number of epochs with which the MLP will be trained.
        thetaOne,thetaTwo  : np.array([][], type = float64), default = None
            Weights of our MLP. Despite being a status, they can be inserted
            by the user as a mean of testing customizations.
        randomize : bool, default = 0
            Defines if the user wants to randomize or not the data.
        
        """
        # Shuffle the data or not utilizing Algorithm.random_ini()
        if randomize:
            inputData, outputData = self.random_ini(self.x, self.y)
        else:
            inputData, outputData = self.x, self.y
        # Slicing the data to the desired train size.
        self.trainInput,self.testInput,self.trainOutput,self.testOutput = self.dataPartition(inputData, outputData,trainSize)
        # To reduce the number of .T operations, do this beforehand.
        self.trainOutput,self.testOutput  = self.trainOutput.T,self.testOutput.T
        
        # Some status definitions.
        [self.p,self.n] = self.trainInput.shape
        self.epochsNum = epochsNum
        self.JList = list()
        
        # The backpropagation Loop.
        for ep in range(0,epochsNum):
            # Feedfowarfing our MLP
            a1, z2, a2, a3 = self.feedf(self.trainInput)
            # Calculating the new Cost (error)
            self.JList += self.costf(self.trainOutput, a3)
            # Calculating the theta using backpropagation
            self.thetaOne, self.thetaTwo = self.backp(self.trainOutput, a3, a2, z2, a1)
            
        return self.thetaOne, self.thetaTwo
                   

    def backp(self, outputData, a3, a2, z2, a1, thetaOne=None, thetaTwo=None):
        """Training our MLP.
        
        In the same way as before, I opted to maintain the thetas as optional
        parameters, so the user can insert custom values to test scenarios.
        
        Parameters
        ----------
        trainSize : float64
            The factor that corresponds to what % of the input data corresponds
            to the train part. 
            ex. trainData = inputData[0:trainSize*inputData.shape(0)]
        epochsNum : int32
            Number of epochs with which the MLP will be trained.
        thetaOne,thetaTwo  : np.array([][], type = float64), default = None
            Weights of our MLP. Despite being a status, they can be inserted
            by the user as a mean of testing customizations.
        randomize : bool, default = 0
            Defines if the user wants to randomize or not the data.
            
        Returns
        ---------
        thetaOne,thetaTwo :  np.array([][], type = float64)
            The new calculated values of thetaOne and thetaTwo
        
        """
        # If the user don't insert Theta, we should get the status one.
        thetaOne = thetaOne if thetaOne is not None else self.thetaOne
        thetaTwo = thetaTwo if thetaTwo is not None else self.thetaTwo
        
        delta3 = a3 - outputData
        thetaTwoBuffer = thetaTwo[:,1:]
        delta2 = (thetaTwoBuffer.T @ delta3)*sigmoid_grad(z2)
        
        thetaTwoGrad = (1/self.p) * (delta3 @ a2.T)
        thetaOneGrad = (1/self.p) * (delta2 @ a1.T)
        thetaTwoGrad[:,1:] += (self.rp/(self.p)) * thetaTwo[:,1:]
        thetaOneGrad[:,1:] += (self.rp/(self.p)) * thetaOne[:,1:]
        thetaOne = thetaOne - self.lrp*thetaOneGrad
        thetaTwo = thetaTwo - self.lrp*thetaTwoGrad
        return thetaOne, thetaTwo
    
    
    def costf(self, outputData, a3, thetaOne=None, thetaTwo=None):
        """Calculating the cost function.
        
        In the same way as before, I opted to maintain the thetas as optional
        parameters, so the user can insert custom values to test scenarios.
        
        Parameters
        ----------
        outputData
        a3 : np.array([][], type = float64)
            The outuput values after our MLP has been feedfoward
        thetaOne,thetaTwo  : np.array([][], type = float64), default = None
            Weights of our MLP.
        
        Returns
        ---------
        J : float64
            The value of the cost given an output value and an expected value.
        """
        thetaOne = thetaOne if thetaOne is not None else self.thetaOne
        thetaTwo = thetaTwo if thetaTwo is not None else self.thetaTwo
        
        J = (1/self.p) * np.sum( np.sum( (-outputData) * np.log(a3)-
                                      (1 - outputData) * np.log(1 - a3))
                                )
        JReg = (self.rp/(2*self.p))*(np.sum(
                np.sum(thetaOne[:,1:]**2)) + 
                np.sum(np.sum(thetaTwo[:,1:]**2)))
        J += JReg      
        return [J]


    def feedf(self, inputData, thetaOne = None, thetaTwo = None):
        """Feedfowarding our MLP.
        
        In the same way as before, I opted to maintain the thetas as optional
        parameters, so the user can insert custom values to test scenarios.
        
        Parameters
        ----------
        outputData
        inputData : np.array([][], type = float64)
            A array containing inputs to execute one interaction.
        thetaOne,thetaTwo  : np.array([][], type = float64), default = None
            Weights of our MLP.
        
        Returns
        ---------
        a1, a2, a3 : np.array([][], type = float64)
            The outputs for each layer (Input, Hidden and Output).
        z2 : np.array([][], type = float64)
            The output of the Hidden layerbefore aplying the activation 
            fucntion.
        """
        thetaOne = thetaOne if thetaOne is not None else self.thetaOne
        thetaTwo = thetaTwo if thetaTwo is not None else self.thetaTwo
        
        a1 = np.concatenate((np.ones((1,self.p)), inputData.T),0)
        z2 = thetaOne @ a1
        a2 = np.concatenate((np.ones((1,self.p)), sigmoid(z2)),0)
        z3 = thetaTwo @ a2
        a3 = sigmoid(z3)
        return a1, z2, a2, a3

    
    def test(self):
        """Evaluanting our MLP.
       
        This method only utilizes information given in the previous methods.
        The TrainData has already been selected in the training.
        
        Returns
        ---------
        cost : float64
            Deprecated
        confArray : np.array([][], type = float64)
            Confusing matrix outputed by the Algorithm.predict method
        """
        [self.p,self.n] = self.testInput.shape
        _,_,_,a3 = self.feedf(self.testInput)
        cost = self.costf(self.testOutput, a3)
        confArray = self.predict(a3.T,self.testOutput.T)
        return cost,confArray
    
    
    def reset(self):
        """Reset the trained MLP to its initial state."""
        
        self.__init__(self.inputLayer, self.hiddenLayer,
                      self.outputLayer)
        return None

    def costplot(self):
        """Plot all the Costs obtained during testing."""
        
        JArray = np.array(self.JList)
        plt.plot(np.arange(len(self.JList)),self.JList)
        plt.axis([0, self.epochsNum, 0, JArray.max()])
        plt.ylabel('Cost function')
        plt.xlabel('No. of Epochs')
        plt.show()
        return None
    
    
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
            self.inputData, self.outputData = self.random_ini(
                                                  self.inputData, 
                                                  self.outputData)
            
        self.trainData,self.testData,_,self.y = self.dataPartition(
                                                    self.inputData,
                                                    self.outputData,
                                                    trainSize)
        S = np.linalg.inv(np.cov(self.trainData.T))
        try:
            self.pred = self.nearestNeighbors(self.trainData, 
                                              self.testData, 
                                              self.y, dist, S)
        except:
            raise 
            
        self.confArray = self.predict(self.pred,self.y)
        return None
      
    
    def nearestNeighbors(self, trainData, testData, y, dist, S):
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
                listBuffer.append(self.mahalanobisDist(arr, trainData, S))
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
        return eDist
    
    
    def mahalanobisDist(self,x1, x2, S):
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
        """"""
        
        self.centroidNum = centroidNum
        self.centroid = np.random.rand(self.centroidNum,
                                       self.inputData.shape[1])
        
        if randomize:
            self.inputData, self.outputData = self.random_ini(self.inputData, 
                                                              self.outputData)
            
        self.trainIn,self.testIn,_,self.eOut = self.dataPartition(
                                                    self.inputData,
                                                    self.outputData,
                                                    trainSize)
        self.nN = 1
        error = 1
        intNum = 0
        S = np.linalg.inv(np.cov(self.trainIn.T))
        
        while np.any(error > 0.00001) or intNum > 10000:
            self.winners = self.cluster_assigment(self.trainIn,self.centroid,dist,S)
            newCentroid = self.move_centroid(self.trainIn, self.winners)
            error = abs(newCentroid - self.centroid)
            self.centroid = newCentroid
            intNum += 1
        self.winners = self.cluster_assigment(self.trainIn,self.centroid,dist,S)
        return self.centroid, self.winners
    
    def cluster_assigment(self, trainData, testData, dist, S):
        """"""
        
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
        outputData = self.eOut
        inputData = self.testIn
        S = np.linalg.inv(np.cov(self.testIn.T))
        winner = self.cluster_assigment(inputData,centroid,'eDist', S)
        pred = np.zeros_like(outputData)
        pred[np.arange(outputData.shape[0]),winner.ravel()] = 1
        confArray = self.predict(pred,outputData)
        self.testOut = winner
        return confArray
    
    
if __name__ == "__main__":
    pass


        
        