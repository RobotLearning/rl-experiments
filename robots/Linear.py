'''
Created on Dec 16, 2016

@author: okan
'''

import numpy as np
from RL.Policies import LinearPolicy 

class Linear(object):
    '''
    Discrete Linear class for testing simple Reinforcement Learning algorithms.
    
    Implemented using state space models.
    '''
    
    def __init__(self, dims, eps, models):
        '''
        Constructor
        '''
        self.dims = dims
        self.noise = eps
        self.x = np.zeros(dims['x'])
        self.models = models
        #self.dt = h
        
    def init_state(self, x):
        '''
        Initialize state to x value.
        '''
        assert(len(x) == self.dims['x'])
        self.x = np.array(x) 
    
    def evolve_state(self, u):
        '''
        Evolves state by integrating one step
        Model matrices are for discrete dynamics
        so no integration necessary
        
        '''
        A = self.models['A']
        B = self.models['B']
        s2 = self.noise['process']
        self.x = np.dot(A,self.x) + np.dot(B,u)
        if np.amax(s2) > 0:
            self.x += np.dot(np.linalg.cholesky(s2),np.random.randn(self.dims['x']))
        
    def observe_state(self):
        
        C = self.models['C']
        y = np.dot(C,self.x)
        s2 = self.noise['observation']
        if np.amax(s2) > 0: 
            y += np.dot(np.linalg.cholesky(s2), \
                    np.random.randn(self.dims['y']))
        
        return y
        
    def rollout(self, policy, x0 = None, max_steps = 50):
        '''
        Generate an episodic trial using the last policy parameters.
        Returns the trajectory tau = [Y,U] state and action sequences.
        
        Policy is given as a lambda function that takes in x 
        and outputs y
        
        Useful for Reinforcement Learning
        '''
        
        Y = np.zeros((self.dims['y'],max_steps+1))
        U = np.zeros((self.dims['u'],max_steps))
        if x0 is not None:
            self.init_state(x0)
        else:
            self.x = np.zeros(self.dims['x'])
            
        for t in range(max_steps):
            y = self.observe_state()
            u = policy.feedback(y,t)
            self.evolve_state(u)
            Y[:,t] = y
            U[:,t] = u
        
        Y[:,-1] = self.observe_state()
        return Y, U
    
def dynamic_riccati(Pnext, Q, R, A, B):
    M = np.dot(Pnext,B)
    N = np.dot(A.transpose(),M)
    S = Q + np.dot(A.transpose(),np.dot(Pnext,A))
    T = R + np.dot(B.transpose(),M)
    if np.size(T) == 1:
        return S - (1/T)*N*N.transpose()
    else:
        return S - N*np.linalg.solve(T,N.transpose())


def iterate_riccati(Qf, Q, R, A, B, N):
    P = Qf
    for i in range(N):
        P = dynamic_riccati(P,Q,R,A,B)
    return P

def compute_init_lqr(Qf,Q,R, A,B,N):
    P = iterate_riccati(Qf,Q,R,A,B,N)
    K = np.dot(B.transpose(),np.dot(P,B))
    if np.size(K) == 1:
        K = -(1/K) * np.dot(B.transpose(),np.dot(P,A))
    else:
        K = -np.linalg.solve(K, np.dot(B.transpose(),np.dot(P,A)))
    return K

def test_linear():
    '''
    Test function for testing linear systems.
    '''
    
    dimx = 2
    dimy = 2
    dimu = 2
    xdes = np.array([0,0])
    x0 = np.array([2,1])
    N = 10
    # dimensions of the linear system, state, observation and control resp.
    dims = {'x': dimx, 'y': dimy, 'u': dimu}
    # noise covariance, process and measurement (observation) respectively
    eps = {'observation': 0.0*np.eye(dimy), 'process': 0.0*np.eye(dimx)}
    # model matrices, A, B and C (no D given)
    a = 0.9
    A = np.array([[0, 1], \
                  [-a*a, -2*a]])
    B = np.eye(2) #np.array([0, 1])
    C = np.eye(2) #np.array([1, 0])
    models = {'A': A, 'B': B, 'C': C}
    # initialize the linear model class
    lin = Linear(dims, eps, models)
    
    # compute initial LQR policy
    Qf = np.eye(2)
    Q = np.zeros((2,2)) #np.eye(2)
    R = np.zeros(1)
    K = compute_init_lqr(Qf, Q, R, A, B, N)
    
    # create a policy    
    # var_policy = 0.1*np.eye(dimu)
    features = lambda x: (x - xdes)
    policy = LinearPolicy(1,features,var = None,theta0 = K)
    # only a final cost/reward
    reward = lambda x,u: -np.dot((x[:,-1]-xdes),np.dot(Qf,x[:,-1]-xdes))
    
    Y, U = lin.rollout(policy,x0,N)
    print('State evolution:') 
    print(Y)
    print('Feedback matrix:')
    print(K)
    print('Reward is %f' % reward(Y,U))
    
if __name__ == "__main__":
    print('Testing linear class with a 2d system...')
    test_linear()    