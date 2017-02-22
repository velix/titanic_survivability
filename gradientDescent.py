import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha):


    # Initialize some useful values
    old_J = computeCost(X, y, theta)
    new_J = 0

    counter = 0
    while abs((old_J - new_J)) > 0.1 :

        hypothesis = np.dot(X, theta)

        loss = hypothesis - y

        X_trans = np.transpose(X)
        gradient = np.dot(X_trans, loss) / X.shape[0]

        theta = theta - alpha*gradient

        old_J = new_J
       
        new_J = computeCost(X, y, theta)
        counter += 1

    return theta

