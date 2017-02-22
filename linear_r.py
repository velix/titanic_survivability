import numpy as np
import pandas as pd
from featureNormalize import featureNormalize
from gradientDescent import gradientDescent

def fit(X, Y, alpha = 0.1):

    X, mu, sigma = featureNormalize(X)
    
    # Init Theta according to dimensions
    theta = np.zeros(X.shape[1])

    theta= gradientDescent(X, Y, theta, alpha)


    return theta
