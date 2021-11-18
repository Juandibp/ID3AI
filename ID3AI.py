import pandas as pd
import numpy as np

trainData = pd.read_csv("data/trainingData.csv")
#print(train_data)

trainData.head()

def calcTotalEntropy(trainData, label, classList):
    totalRow = trainData.shape[0] #total size of dataset
    totalEntropy = 0

    for c in classList: #for each class in label
        totalClassCount = trainData[trainData[label] == c].shape[0] #number of class
        totalClassEntropy = -(totalClassCount/totalRow)*np.log2(totalClassCount/totalRow) #entropy of class
        totalEntropy += totalClassEntropy # add class entropy to total
    
    return totalEntropy


def calcEntropy(featuredValueData, label, classList):
    classCount = featuredValueData.shape[0]
    entropy = 0

    for c in classList:
        labelClassCount = featuredValueData[featuredValueData[label]==c].shape[0] # row count of class c
        entropyClass = 0
        if labelClassCount != 0:
            probabilityClass = labelClassCount/classCount #prob of class
            entropyClass = -probabilityClass * np.log2(probabilityClass) #calc entropy
        entropy += entropyClass
    
    return entropy