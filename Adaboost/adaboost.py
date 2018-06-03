#encoding=utf-8
'''
Created on 2018年6月2日

@author: Administrator
'''
import numpy as np

def loadSimpData():
    '''
    create simple dataset
    return: dataMat(np.mat) classLabels(array)
    '''
    dataMat=np.mat([[1,2.1],[2,1.1],[1.3,1],[1,1],[2,1]]);
    classLabels=[1.0,1.0,-1.0,-1.0,1.0];
    return dataMat,classLabels;

def loadDataSet(filename):
    '''
    load dataset
    return:(mat)data matrix and (mat)label matrix
    '''
    numFeat=len(open(filename).readline().split("\t"));
    dataMat=[];
    labelMat=[];
    fr=open(filename,"r");
    for line in fr.readlines():
        currArr=[];
        lineArr=line.strip().split("\t");
        for i in range(numFeat-1):
            currArr.append(lineArr[i]);
        dataMat.append(currArr);
        labelMat.append(lineArr[-1]);
    return np.mat(dataMat).astype("float"),np.mat(labelMat).astype("float");

def stumpClassify(dataMat,dimen,threshVal,threshIneq):
    '''
    set the binary classification function
    dataMat: (np.mat)input data
    dimen: (int)the axis of X
    threshVal: (double)threshold used to classify data
    threshIneq: (string)inequal(less than or great than)
    return: (array)classified result
    '''
    retArray=np.ones((dataMat.shape[0],1));
    if threshIneq=="lt":
        retArray[dataMat[:,dimen]<=threshVal]=-1;
    else:
        retArray[dataMat[:,dimen]>threshVal]= 1;
    return retArray;

def buildStump(dataArr,classLabels,D):
    '''
    build the decision tree with least error
    dataArr: (array or mat) data input X
    classLabels: (array or mat) data labels Y
    D: (mat) weight mat
    '''
    dataMat=np.mat(dataArr);labelMat=np.mat(classLabels).T;
    m,n=dataMat.shape;
    numSteps=10;
    bestStump={};
    bestClassEst=np.mat(np.zeros((m,1)));
    minError=10000;
    
    for i in range(n):
        #iterate in each feature
        rangeMin=dataMat[:,i].min();
        rangeMax=dataMat[:,i].max();
        stepSize=(rangeMax-rangeMin)/numSteps;
        for j in range(-1,numSteps+1):
            for inequal in ["lt","gt"]:
                threshVal=rangeMin+j*stepSize;
                predictedVals=stumpClassify(dataMat, i, threshVal, inequal);
                errMat=np.mat(np.ones((m,1)));
                errMat[predictedVals==labelMat]=0;
                weightedError=D.T*errMat;
                #print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f"%(i,threshVal,inequal,weightedError));
                if weightedError<minError:
                    minError=weightedError;
                    bestClassEst=predictedVals.copy();
                    bestStump["dim"]=i;
                    bestStump["thresh"]=threshVal;
                    bestStump["ineq"]=inequal;
    return bestStump,minError,bestClassEst;

def adaBoostTrainDS(dataArr,classLabels,maxIter=40):
    '''
    adaBoost classifier
    dataArr: (array or mat) input data
    classLabels: (array or mat) data labels
    maxIter: (int) max iteration numbers
    return: (list) weak classifiers list
    '''
    weakClassList=[];
    m=dataArr.shape[0];
    D=np.mat(np.ones((m,1))/m);#initialize weight as 1/m
    aggClassEst=np.mat(np.zeros((m,1)));
    labelMat=np.mat(classLabels);
    for i in range(maxIter):
        bestStump,error,classEst=buildStump(dataArr, classLabels, D);
        #print("D: ",end="");print(D[:,0].T);
        alpha=(0.5*np.log((1.0-error)/max(error,1e-16)))[0,0];
        bestStump["alpha"]=alpha;
        weakClassList.append(bestStump);#aggregate week classifiers
        #print("classEst: ",end="");print(classEst.T);
        
        expon=np.multiply(-1*(alpha*labelMat).T,classEst);
        D=np.multiply(D,np.exp(expon));
        D=D/D.sum();
        aggClassEst+=alpha*classEst;
        #print("aggClassEst: ",end="");print(aggClassEst.T);
        aggErrors=np.multiply(np.sign(aggClassEst)!=labelMat.T,np.ones((m,1)));
        errorRate=aggErrors.sum()/m;
        print("total error:%f"%errorRate);
        if errorRate==0.0:
            break;
    return weakClassList;

def adaClassify(dataToClassify,classifierList):
    '''
    dataToclassify: (array) to be classified
    classifierList: (list) already trained classifiers list
    return: classify result
    '''
    dataMat=np.mat(dataToClassify);
    m=dataMat.shape[0];
    aggClassEst=np.mat(np.zeros((m,1)));
    for i in range(len(classifierList)):
        classEst=stumpClassify(dataMat, classifierList[i]["dim"], classifierList[i]["thresh"], classifierList[i]["ineq"]);
        aggClassEst+=classifierList[i]["alpha"]*classEst;
        #print(aggClassEst);
    return np.sign(aggClassEst);
    

if __name__=="__main__":
    dataMat,labelMat=loadDataSet("horseColicTraining2.txt");
    classifierList=adaBoostTrainDS(dataMat, labelMat,100);
    testMat,testLabel=loadDataSet("horseColicTest2.txt");
    predict=adaClassify(testMat, classifierList);
    errorMat=np.mat(np.zeros((testMat.shape[0],1)));
    errorMat[predict!=testLabel.T]=1;
    averError=errorMat.sum()/testMat.shape[0];
    print(averError);
    
    
