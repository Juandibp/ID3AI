import pandas as pd
import numpy as np
import sys
sys.setrecursionlimit(50000)

MainTrainData = pd.read_csv("data/trainingData.csv")
#print(train_data)

MainTrainData.head()

def calcTotalEntropy(trainData, label, classList):
    totalRow = trainData.shape[0] #total size of dataset
    totalEntropy = 0

    for c in classList: #for each class in label
        ##print(trainData[trainData[label] == c])
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


#Calculating information gain for a feature
def calcInfoGain(featureName, trainData, label, classList):
    featureValueList = trainData[featureName].unique() #unique values of the feature
    totalRow = trainData.shape[0]
    featureInfo = 0.0

    for featureValue in featureValueList:
        featureValueData = trainData[trainData[featureName]==featureValue] #filtering rows with that featureValue
        featureValueCount = featureValueData.shape[0]
        featureValueEntropy = calcEntropy(featureValueData, label, classList) #Calculate Entropy for feature value
        featureValueProbability = featureValueCount/totalRow
        featureInfo += featureValueProbability * featureValueEntropy #calculate info of feature value

    return calcTotalEntropy(trainData, label, classList) - featureInfo #calculating info gain by substracting

#Finding the most informative feature (feature with highest information gain)

def findMostInformativeFeature(trainData, label, classList):
    featureList = trainData.columns.drop(label) # finding the feature names in the dataset
    
    maxInfoGain = -1
    maxInfoFeature = None

    for feature in featureList: #for each feature in dataset
        featureInfoGain = calcInfoGain(feature, trainData, label, classList)
        if maxInfoGain < featureInfoGain:
            maxInfoGain = featureInfoGain
            maxInfoFeature = feature

    return maxInfoFeature


#adding node to tree

def generateSubTree(featureName, trainData, label, classList):
    #print(trainData["RangoIngreso"])
    featureValueCountDict = trainData[featureName].value_counts(sort = False) #dictionary of the count of unique feature values
    tree = {}

    for featureValue, count in featureValueCountDict.iteritems():
        featureValueData = trainData[trainData[featureName]==featureValue] #dataset with only featureName = featureValue
        assignedToNode = False # Flag for tracking featureValue is pure class or not

        for c in classList: #for each class
            classCount = featureValueData[featureValueData[label]==c].shape[0] #count of class c

            if classCount == count: #count of feature value = count of class (pure class)
                tree[featureValue] = c #add node to tree
                trainData = trainData[trainData[featureName] != featureValue] #removing rows with feature value
                assignedToNode = True

        if not assignedToNode: # not pure class
            tree[featureValue] = "?" #should extend the node so iots marked

    return tree, trainData


#Performing ID3 Algorithm and generating Tree
def makeTree(root, prevFeatureValue, trainData, label, classList):
    if trainData.shape[0] !=0: #if dataset is not empty after updating
        maxInfoFeature = findMostInformativeFeature(trainData, label, classList)
        tree, trainData = generateSubTree(maxInfoFeature, trainData, label, classList)
        #print(tree)
        nextRoot = None

        if prevFeatureValue != None: #add to intermediate node of tree
            root[prevFeatureValue] = dict()
            root[prevFeatureValue][maxInfoFeature] = tree
            nextRoot = root[prevFeatureValue][maxInfoFeature]
        else: #add to root of the tree
            root[maxInfoFeature] = tree
            nextRoot = root[maxInfoFeature]

        for node, branch in list(nextRoot.items()):
            if branch == "?": #if its expandable
                #print('expand dong')
                featureValueData = trainData[trainData[maxInfoFeature]==node] #using updated dataset
                makeTree(nextRoot, node, featureValueData, label, classList) #recursive call with new dataset

#Here goes Nothing, havent tested anything

def id3(MainTrainData, label):
    trainData = MainTrainData.copy()
    #print(trainData['RangoIngreso'])
    tree = {}
    classList = trainData[label].unique()
    makeTree(tree, None, MainTrainData, label, classList)
    return tree

tree = id3(MainTrainData,"Deuda")
print(tree)