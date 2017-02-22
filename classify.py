import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from generateTree import generateTree
#from sklearn import neighbors
from sklearn import tree
from kNN import kNN
#import linear_r as lr

def classify_randomForest(trainSet, trainLabels, testSet):

    predictedLabels = np.zeros(testSet.shape[0])

    #criterion: default 'gini' TODO: check with 'entropy'
    ######
    #criterion = 'gini': 0.8092
    #criterion = 'entropy': 0.8181
    #n_estimators: ???
    #10 50 100
    forest = RandomForestClassifier(n_estimators = 100, criterion = 'entropy')

    forest = forest.fit(trainSet, trainLabels)

    predictedLabels = forest.predict(testSet)

    #predictedLabels = predictedLabels.tolist()

    return predictedLabels



def classify_kNN(trainSet, trainLabels, testSet):
    # k = 5 best accuracy
    n_neighbors = 5
    predictedLabels = []

    for i in range(testSet.shape[0]):
        val = kNN(n_neighbors, trainSet, trainLabels, testSet[i,:])
        predictedLabels.append(val)
    
   

    return predictedLabels

def classify_decisionTree(trainSet, trainLabels, testSet):

    # we create an instance of Neighbours Classifier and fit the data.
    #clf = tree.DecisionTreeClassifier(criterion = 'gini')
    clf = tree.DecisionTreeClassifier(criterion = 'entropy')

    clf = clf.fit(trainSet, trainLabels)

    predictedLabels = clf.predict(testSet)

    return predictedLabels

