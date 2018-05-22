#encoding=utf-8
'''
Created on 2018年5月22日

@author: Administrator
'''
import numpy as np
import math
import random

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']];
    classVec = [0,1,0,1,0,1];    #1表示侮辱类，0表示不属于
    return postingList,classVec; #词条切分后的分档和类别标签

def createVocabList(dataSet):
    '''
    create vocabulary list 
    ignore the frequency of each word
    '''
    vocabSet=set([]);
    for document in dataSet:
        vocabSet=vocabSet|set(document);
    return list(vocabSet);

def setOfWords2Vec(vocabList,inputSet):
    '''
    transfer word to vector by recording the word index
    return vector of inputSet
    '''
    returnVec=[0]*len(vocabList);
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1;
        else:
            print("the word: %s is not in the word list"%(word));
    return returnVec;

def bagOfWords2Vec(vocabList,inputSet):
    '''
    transfer word to vector by recording the word index and statistic frequency
    vocabList: vocabulary list storing all words
    inputSet: to be transfered word set
    return vector of inputSet
    '''
    returnVec=[0]*len(vocabList);
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1;
        else:
            print("the word: %s is not in the word list"%(word));
    return returnVec;

def trainNB0(trainMat,trainCategory):
    '''
    calculate probability of each category and the frequency of each word.
    trainMat: dataset made of vectors
    trainCategory: predefined classes corresponding to each record in trainMat
    return: p(w|c=0),p(w|c=1),p(c=1)
    '''
    numTrainDocs=len(trainMat);
    numWords=len(trainMat[0]);
    pc_1=sum(trainCategory)/numTrainDocs;
    #p0Norm=0.1 to avoid overflow
    p0Num=np.ones(numWords);p0Norm=0.1;
    p1Num=np.ones(numWords);p1Norm=0.1;
    for i in range(numTrainDocs):
        #if class = 1
        if trainCategory[i]==1:
            p1Num+=trainMat[i];
            p1Norm+=sum(trainMat[i]);
        #if class = 0
        else:
            p0Num+=trainMat[i];
            p0Norm+=sum(trainMat[i]);
            
    #use ln() to replace following multiply and avoid underflow
    p1Vec=np.log(p1Num/p1Norm);
    p0Vec=np.log(p0Num/p0Norm);
    return p0Vec,p1Vec,pc_1;

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''
    classification
    vec2Classify: to be classified vector
    p0Vec: p(w|c=0)
    p1Vec: p(w|c=1)
    pClass1: p(c=1)
    return: predicted class
    '''
    p1=sum(vec2Classify*p1Vec)+math.log(pClass1);
    p0=sum(vec2Classify*p0Vec)+math.log(1-pClass1);
    if p1>p0:
        return 1;
    elif p1<p0:
        return 0;
    else:
        print("unknown class!");

def textParse(bigStr):
    '''
    parse email text
    bigStr:e-mail text
    return: word list
    '''
    import re
    strList=re.split("\\W*",bigStr);
    return [word.lower() for word in strList if len(word)>2];

if __name__=="__main__":
    postList,labelList=loadDataSet();
    vocabList=createVocabList(postList);
    trainMat=[];
    for i in range(len(postList)):
        wordVec=setOfWords2Vec(vocabList, postList[i]);
        trainMat.append(wordVec);
    p0,p1,pc1=trainNB0(trainMat, labelList);
    textEntry=["stupid","garbage"];
    thisDoc=bagOfWords2Vec(vocabList, textEntry);
    classifiedResult=classifyNB(thisDoc, p0, p1, pc1)
    print("the classified result of %s is class %d"%(textEntry[:],classifiedResult));