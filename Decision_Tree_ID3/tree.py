#encoding=utf-8
'''
Created on 2018年5月21日

@author: zk
'''
import math
import pickle
from chapter3 import treePlotter
import operator

def calcShannonEntropy(dataSet):
    #calculate shannon entropu of dataSet
    
    numEntries=len(dataSet);
    labelCounts={};
    for featVec in dataSet:
        label=featVec[-1];
        if label not in labelCounts.keys():
            labelCounts[label]=0;
        labelCounts[label]+=1;
    shannonEntropy=0.0;
    for key in labelCounts.keys():
        prob=float(labelCounts[key])/numEntries;
        shannonEntropy-=prob*math.log2(prob);
    return shannonEntropy;
def createDataSet():
    dataSet=[[1,1,"yes"],[1,1,"yes"],[1,0,"no"],[0,1,"no"],[0,1,"no"]];
    labels=["no surfacing","flippers"];
    return dataSet,labels;
def splitDataSet(dataSet,axis,value):
    #split dataSet by axis and return remaining elements
    
    retDataSet=[];
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis];
            reducedFeatVec.extend(featVec[axis+1:]);
            retDataSet.append(reducedFeatVec);
    return retDataSet;

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1;
    baseEntropy=calcShannonEntropy(dataSet);
    bestInfoGain=0.0;
    bestFeature=-1;
    for i in range(numFeatures):
        featList=[example[i] for example in dataSet];
        #use set to get unique values from feature list
        uniqueVals=set(featList);
        newEntropy=0.0;
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet, i, value);
            prob=len(subDataSet)/float(len(dataSet));
            newEntropy+=prob*calcShannonEntropy(subDataSet);
        infoGain=baseEntropy-newEntropy;
        if(infoGain>bestInfoGain):
            bestInfoGain=infoGain;
            bestFeature=i;
    return bestFeature;

def majorityCount(classList):
    classCount={};
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0;
        classCount[vote]+=1;
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True);
    return sortedClassCount[0][0];
def createTree(dataSet,labels):
    '''
    generate decision tree
    '''
    classLabels=labels[:];
    classList=[example[-1] for example in dataSet];
    #if all classes are same in this list
    if classList.count(classList[0])==len(dataSet):
        return classList[0];
    #if all features are assigned
    if len(classList[0])==1:
        return majorityCount(classList);
    bestFeature=chooseBestFeatureToSplit(dataSet);
    bestFeatLabel=classLabels[bestFeature];
    myTree={bestFeatLabel:{}};
    del(classLabels[bestFeature]);
    featValue=[example[bestFeature] for example in dataSet];
    uniqueVal=set(featValue);
    for value in uniqueVal:
        sublabels=classLabels[:];
        subDataSet=splitDataSet(dataSet, bestFeature, value);
        myTree[bestFeatLabel][value]=createTree(subDataSet, sublabels);
    return myTree;

def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0];
    secondDict=inputTree[firstStr];
    featIndex=featLabels.index(firstStr);
    for key in secondDict.keys():
        if key==testVec[featIndex]:
            if type(secondDict[key]).__name__=="dict":
                classLabel=classify(secondDict[key],featLabels,testVec);
            else:
                classLabel=secondDict[key];
    return classLabel;

def storeTree(inputTree,filename):
    fw=open(filename,"wb");
    pickle.dump(inputTree,fw);
    fw.close();
    
def loadTree(filename):
    fr=open(filename);
    return pickle.load(fr);

if __name__=="__main__":
    fr=open("lenses.txt");
    lenses=[line.strip() for line in fr.readlines()];
    lenseDataSet=[];
    for line in lenses:
        dataList=line.split("\t");
        lenseDataSet.append(dataList);
    lensesLabels=["age","prescript","astigmatic","tearRate"];
    lensesTree=createTree(lenseDataSet, lensesLabels);
    treePlotter.createPlot(lensesTree);