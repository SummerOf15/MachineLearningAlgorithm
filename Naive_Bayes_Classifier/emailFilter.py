#encoding=utf-8
'''
Created on 2018年5月22日

@author: Administrator
'''
from bayes import *

def spamTest():
    '''
    determine whether a text is a spam
    '''
    docList=[];classList=[];fullWordList=[];
    for i in range(1,26):
        spamMailText=open("email/spam/%s.txt"%i,"rb").read();
        wordList=textParse(spamMailText.decode("gbk"));
        docList.append(wordList);
        fullWordList.extend(wordList);
        classList.append(1);#1 means spam email
        hamMailText=open("email/ham/%s.txt"%i,"rb").read();
        wordList=textParse(hamMailText.decode("gbk"));
        docList.append(wordList);
        fullWordList.extend(wordList);
        classList.append(0);#0 means ham
    vocabList=createVocabList(docList);
    trainingSet=[i for i in range(50)];#there are totally 50 emails
    testSet=[];
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)));
        testSet.append(trainingSet[randIndex]);
        del(trainingSet[randIndex]);
    trainMat=[];
    trainClasses=[];
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]));
        trainClasses.append(classList[docIndex]);
    p0Vec,p1Vec,pSpam=trainNB0(trainMat, trainClasses);
    errorCount=0;
    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList, docList[docIndex]);
        if classifyNB(wordVector, p0Vec, p1Vec, pSpam)!=classList[docIndex]:
            errorCount+=1;
    return float(errorCount)/len(testSet);

if __name__=="__main__":
    totalErrorRate=0;
    loopNum=10;
    #hold out cross validation
    for i in range(loopNum):
        totalErrorRate+=spamTest();
    print("The average error rate is %f"%(totalErrorRate/loopNum));