import numpy as np


def computeCost(X, y, theta):


    hypothesis = np.dot(X, theta)

    
    #loss = hypothesis - y
    loss = np.subtract(hypothesis, y)


    term = np.transpose(loss)*(loss)

    j = term/2*X.shape[0]


    return np.sum(j, axis = 0)
