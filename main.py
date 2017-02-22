import numpy as np
import pandas as pd
from sklearn import cross_validation
import csv as csv
from classify import *
#import normalize as norm
#import linear_r as lr



#load data
data = pd.read_csv('train.csv', header = 0)

#map the values of Sex to integers on new column Gender
data['Gender'] = data['Sex'].map({'male':0, 'female':1})
data['Room'] = data['Cabin'].map(lambda x: str(x)[0].upper())
data['Room'] = data['Room'].map({'A':1, 'B':2, 'C':3, 'D':4, 'E': 5, 'F': 6, 'G': 7})

for i in range(len(data.Room)):
    if ((data.Room[i] != 1) & (data.Room[i] != 2) & (data.Room[i] != 3) & (data.Room[i] != 4) & (data.Room[i] != 5) & (data.Room[i] != 6) & (data.Room[i] != 7)):
         data.Room[i] = 0


#map the values of Embarked to integers on new column Port
#data['Port'] = data['Embarked'].map({'S':1, 'C':2, 'Q': 3, np.nan : 1})

#delete 
data = data.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Fare', 'Cabin', 'Embarked'], axis = 1)
#data = data.drop(['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis = 1)

print data.columns.values

full_data = data[data['Age'].notnull()]
ages = full_data['Age']
full_data = full_data.drop(['Age'], axis = 1)

full_data = full_data.astype(float)

# print "data\n",data.info()
# print "\n"
# print "full_data\n",full_data.info()
# print "\n"
# print "ages ",np.size(ages)


theta = lr.fit(full_data.as_matrix(), ages.as_matrix())[0]
nan_age_indices = np.where(pd.isnull(data['Age']))[0]


data_no_age = data.drop(['Age'], axis = 1)

#on data.Age as indexed by the indices where age is nan, 
#assign the dot product of the rest of the particular instance with theta
data['Age'].iloc[nan_age_indices] = np.dot(data_no_age.iloc[nan_age_indices], theta)

#4 instances with negative, same (-0.176445) age :(
negative_age_ind = np.where(data['Age'] <= 0)
#replace negative age with average age
data.Age.iloc[negative_age_ind] = np.sum(data['Age'], axis = 0)/np.size(data['Age'])


#-----------------------------------
#find where the fare is nan or 0
#replace those instances with the mean fare
# na_fare_ind = np.where(data['Fare'] == 0)
# data.Fare.iloc[na_fare_ind] = np.sum(data['Fare'], axis = 0)/np.size(data['Fare'])


# full_data = data[data['Room'].notnull()]
# rooms  = full_data['Room']
# full_data = full_data.drop(['Room'], axis = 1)

# full_data = full_data.astype(float)

# theta = lr.fit(full_data.as_matrix(), rooms.as_matrix(), 0.0001)[0]
# nan_room_indices = np.where(pd.isnull(data['Room']))[0]

# data_no_room = data.drop(['Room'], axis = 1)

# data['Room'].iloc[nan_room_indices] = np.dot(data_no_age.iloc[nan_room_indices], theta)

# print data.Room.values



data['Child'] = 0
for i in range(len(data.Age)):
	if data.Age[i] <= 8:
		data['Child'][i] = 5
	else:
		if data.Gender[i]=='male':
			data['Child'][i] = 1
		else:
			data['Child'][i] = 9

#keep in y if survived
y = data['Survived']

data = data.drop('Survived', axis = 1)

data = data.as_matrix()
y = y.as_matrix()

#Initialize cross validation
kf = cross_validation.KFold(data.shape[0], n_folds=10)

totalInstances = 0 # Variable that will store the total intances that will be tested  
totalCorrect = 0 # Variable that will store the correctly predicted intances  

for trainIndex, testIndex in kf:
    trainSet = data[trainIndex]
    testSet = data[testIndex]
    trainLabels = y[trainIndex]
    testLabels = y[testIndex]
    
    #predictedLabels = classify_kNN(trainSet, trainLabels, testSet) #k = 5
    #predictedLabels = classify_decisionTree(trainSet, trainLabels, testSet)
    predictedLabels = classify_randomForest(trainSet, trainLabels, testSet)

   
    correct = 0   
    for i in range(testSet.shape[0]):

        if predictedLabels[i] == testLabels[i]:
            correct += 1
        
    print '\nAccuracy: ' + str(float(correct)/(testLabels.size))
    totalCorrect += correct
    totalInstances += testLabels.size
print 'Total Accuracy: ' + str(totalCorrect/float(totalInstances))