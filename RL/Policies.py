'''

Keeping policy classes here.

#TODO: Implementing so far only linear policy, gaussian likelihood and
       fixed variance

Created on Dec 19, 2016

@author: okan
'''
import numpy as np

class LinearPolicy(object):
    '''
    Linear policy class that is a linear combination of features.
    
    Features are phi(x), given in the initialization.
    '''
    
    def __init__(self, dim, features, var = None, theta0 = None):
        '''
        Policy is a linear combination of features.
        Features are given as function of state variables.
        
        Variance is fixed for the Gaussian distribution (stochastic policy).
        '''
        
        self.dim = dim # dimension of the features
        self.features = features
        if theta0 is not None:
            self.theta = theta0
        else:
            self.theta = np.zeros(dim) 
            
        self.var = var
        
    def calc_log_der(self, u, x, t):
        '''
        Calculates the derivative of the log likelihood
        for the implemented policy.
        '''
        val = np.linalg.solve(self.var, \
                        u - np.dot(self.theta,self.features(x)))
        return self.features(x) * val


    def feedback(self, x, t):
        '''
        Apply feedback at time step t.
        
        If covariance of policy is not initialized as None
        then we add a gaussian noise to the control inputs at time step t.
        
        '''
        
        dimu = 1
        u = np.dot(self.theta,self.features(x))
        if self.var is not None:
            noise = np.random.randn(dimu)
            u += np.dot(np.linalg.cholesky(self.var),noise)
                       
        return u