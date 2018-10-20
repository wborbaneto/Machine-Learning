# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:19:17 2018

@author: AHCI
"""
import numpy as np


def out_b(slices):
    """Create an binary array to represent classes.
    
    outputb([1,2,1]) : ans =    [1,0,0]
                                [0,1,0]
                                [0,1,0]
                                [0,0,1]
    Parameters
    ----------
    slices : list(type = int32)
        Number of rows for each class.
        
    Returns
    ----------
    y : np.array([][], type = int32)
        The desired binary array.
    """
    
    numC = len(slices)
    y = np.zeros([1,numC])
    for s in range(0,numC):
        buffer = np.zeros([1, numC])
        buffer[:,s] = 1
        y = np.r_[ y,np.tile(buffer,(slices[s],1))]
    return y[1:,:]

def out_d(slices):
    """Create an decimal array to represent classes.
    
    outputb([1,2,1]) : ans =    [0]
                                [1]
                                [1]
                                [2]
    Parameters
    ----------
    slices : list(type = int32)
        Number of rows for each class.
        
    Returns
    ----------
    y : np.array([][], type = int32)
        The desired binary array.
    """
    numC = len(slices)
    y = np.zeros([1,1])
    for s in range(0,numC):
        y = np.r_[ y,np.tile(s,(slices[s],1))]
    return y[1:,:]

def out_s(slices, names):
    """Create an string array to represent classes.
    
    outputb([1,2,1], ['a','b','c']) : ans = ['a']
                                            ['b']
                                            ['b']
                                            ['c']
    Parameters
    ----------
    slices : list(type = int32)
        Number of rows for each class.
    names : list(type = U32)
        Names given to each class.
        
    Returns
    ----------
    y : np.array([][], type = int32)
        The desired binary array.
    """
    numC = len(slices)
    y = np.zeros([1,1])
    for s in range(0,numC):
        y = np.r_[ y,np.tile(names[s],(slices[s],1))]
    return y[1:,:]

def dec2str(decimal, names):
    """Converts a decimal array in a string array.
    
    outputb([0,1,1,2], ['a','b','c']) : ans = ['a']
                                              ['b']
                                              ['b']
                                              ['c']
    Parameters
    ----------
    decimal : np.array([][], type = int32)
        Decimal array.
    names : list(type = U32)
        Names given to each class.
        
    Returns
    ----------
    y : np.array([][], type = int32)
        The desired binary array.
    """
    numC = len(names)
    y = np.tile("               ",(decimal.shape))
    for s in range(0,numC):
        y[(decimal==s).ravel()] = names[s]
    return y

def bin2str(binary, names):
    """Converts a binary array in a string array.
    
    outputb([[1,0,0],[0,1,0],[0,1,0],[0,0,1]], ['a','b','c']) 
    ans =   ['a']
            ['b']
            ['b']
            ['c']
            
    Parameters
    ----------
    binary: np.array([][], type = int32)
        Binary array.
    names : list(type = U32)
        Names given to each class.
        
    Returns
    ----------
    y : np.array([][], type = int32)
        The desired binary array.
    """
    numC = len(names)
    y = np.tile("               ",(binary.shape[0],1))
    for s in range(0,numC):
        buffer = np.zeros([1, numC])
        buffer[:,s] = 1
        y[np.all(binary==buffer,1).ravel()] = names[s]
    return y