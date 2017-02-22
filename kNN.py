from numpy import *
from euclideanDistance import euclideanDistance

def kNN(k, X, labels, y):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
	# training examples using the kNN with euclidean distance.

    m = X.shape[0] # number of training examples            number of Rows
    n = X.shape[1] # number of attributes                   number of columns

    closest = zeros((k,2)) # stores the k closest to the test instance training examples
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Run the kNN algorithm to predict the class of
    #				y. Rows of X correspond to observations, columns
    #               to features. The labels vector contains the 
    #               class to which each observation belongs. k 
    #               corresponds to the parameter k of KNN. Calculate
    #               the distance betweet y and each row of X, find 
    #               the k closest observations and give y the class
    #               of the majority of them.
    #
    # Note: To compute the distance betweet two vectors A and B use
    #		use the euclideanDistance(A,B) function.
    #

    #Below is some bad coding yo
    
    distances = zeros((m, 2))
    for i in range(m):
        #the first column has the distances
        distances[i,0] = euclideanDistance(y, X[i,:])
        # ont the second column stick the labels
        distances[i,1] = labels[i] 


    #i want to sort the goddamn distances on the first column. the one witht the distances
    distances = distances[distances[:,0].argsort()]
   


    #or probably this, to get the k closest. (it actually returns up to k-1 but it's indexed on zero so...)
    closest = distances[:k]

    # print "closesr are:\n",closest
    # print"-"*10

    #dictionary with labels and occurences
    lbls = {}
    for i in range(k):
        lbls[distances[i,1]] = 0

    for i in range(k):
        #foo stores the current label
        foo = closest[i,1]

        for j in range(k):
            if(not(i == j)):
                if(closest[j,1] == foo):
                    lbls[foo] = lbls[foo] + 1


    label = max(lbls.iterkeys(), key = (lambda key: lbls[key]))

    #print label


    
	
	
    # =============================================================

    
    return label

 
