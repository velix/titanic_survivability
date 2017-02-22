import numpy as np

def featureNormalize(X): 

# Returns a normalized version of X where
# the mean value of each feature is 0 and the standard deviation
# is 1. 

    #X_norm = X.copy() #copy matrix X to X_norm
    X_norm = np.array(X.copy())

# ====================== YOUR CODE HERE ======================
# Instructions: Calculate the mean value mu of the one-dimensional 
#				matrix X_norm. Subtract the mean value from each
#				entry. Calculate the standard deviation sigma of 
#				X_norm. Divide each entry of X_norm by the standard
#				deviation. For calculating the mean and standard
#				deviation, you can use the NumPy built-in functions 
#				mean(X_norm) and std(X_norm).				


    #calculate mean on axis 0 (columns)
    #probably transpose it??
    mu = np.mean(X_norm, axis = 0)  

     #get standar deviation on axis = 0 (columns)
    sigma = np.std(X_norm, axis = 0)
    
   
    
    (X_norm - mu)/sigma 



# =============================================================
    
    		
    return X_norm, mu, sigma	

