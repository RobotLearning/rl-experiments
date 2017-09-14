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
        
        u = np.dot(self.theta,self.features(x))
        if self.var is not None:
            noise = np.random.randn(len(self.theta))
            u += np.dot(np.linalg.cholesky(self.var),noise)
                       
        return u
    
def test_feedback():    
    '''
    Testing convergence to a desired set position 
    for a critically damped system with feedback and const. feedforward inputs 
    '''
    from robots.Linear import Linear
    
    # dimensions of the linear system, state, observation and control resp.
    dims = {'x': 2, 'y': 1, 'u': 1}
    # noise covariance, process and measurement (observation) respectively
    eps = {'observation': 0.0, 'process': 0.0}  
    
    # settings for the differential equation and pd-control parameters
    a = 0.9 # damping constant
    xdes = 1.0 # desired set point
    kd = 0.1
    kp = kd*kd/4 + a*kd
    kc = -a*a*xdes
    x0 = np.zeros(dims['x'])
    
    A = np.array([[0, 1], \
                  [-a*a, -2*a]])
    B = np.array([[0], [1]])
    C = np.eye(2)
    models = {'A': A, 'B': B, 'C': C}    
    # initialize the linear model class
    lin = Linear(dims, eps, models)
    features = lambda x: np.concatenate(([1],x))
    theta = np.array((kp,kd,kc))
    policy = LinearPolicy(1,features,var = None, theta0 = theta)
    [x,u] = lin.rollout(policy, x0, 10)
    reward = lambda x,u: -(x[:,-1] - xdes)**2
    print('Reward for the trajectory is:', reward(x,u))
    
if __name__ == "__main__":
    print('Testing feedback that makes system critically damped around xdes...')
    test_feedback()    