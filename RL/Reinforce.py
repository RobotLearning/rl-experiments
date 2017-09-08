'''
Created on Dec 17, 2016

Here we implement and test the REINFORCE algorithm,
one of the first Policy Gradient algorithms introduced 
in the literature.

@author: okan
'''

import numpy as np
from RL.Policies import LinearPolicy
from robots.Linear import Linear

class Reinforce(object):
    '''
    Episodic policy gradient based approach.
    '''
    
    def __init__(self, policy, reward, horizon, num_rollouts):
        '''
        Constructor.
        '''
        self.policy = policy
        self.reward = reward
        self.horizon = horizon
        self.rolloutsize = num_rollouts
        
    def learn(self, tol, x0, model, verbose):
        '''
        Typical learning procedure composed of rollouts, estimating gradient,
        and then gradient descent till parameter update is less than given tol.
        
        '''
        
        #TODO: check for parameter limits to ensure stability!
        
        ITER_MAX = 100
        iter_num = 1
        theta_last = self.policy.theta.copy() + 2*tol
        while np.max(np.abs(self.policy.theta - theta_last)) > tol \
                and iter_num < ITER_MAX:
            # do some rollouts
            D = self.rollouts(x0, model)
            # estimate gradient
            grad = self.estimate_gradient(D)
            # do line search and update gradient
            alpha = self.line_search(grad)
            theta_last = self.policy.theta.copy()
            self.policy.theta += alpha * grad
            if verbose:
                print('Iteration %d' %iter_num)
                print('Average reward of trial: %f' %(np.sum(D['rewards'])/self.rolloutsize))
                iter_num += 1
        
    def rollouts(self, x0, model):
        '''
        Make N rollouts on the system 'model' class from initial state x0.
        Each rollout consists of T (maximum) time steps.
    
        Returns the state action sequences and the rewards
        as a dataset D.
        '''
        
        N = self.rolloutsize
        T = self.horizon
        X = np.zeros((model.dims['y'], (T+1)*N))
        U = np.zeros((model.dims['u'], T*N))
        R = np.zeros((N))
        for i in range(N):
            [y,u] = model.rollout(self.policy, x0, T)
            R[i] = self.reward(y,u)
            X[:,i*(T+1):(i+1)*(T+1)] = y
            U[:,i*T:(i+1)*T] = u
        
        D = {'states': X, 'actions': U, 'rewards': R}
        return D
    
    def estimate_baseline(self, D):
        '''
        Estimate the optimal gradient baseline.
        
        Reduces the variance of policy gradient based updates, 
        without putting bias on the estimates.
        '''
        
        N = self.rolloutsize
        T = self.horizon
        dimh = self.policy.dim
        R = D['rewards']
        U = D['actions']
        X = D['states']
        baseline = np.zeros(dimh)
        
        # estimate the baseline of policy for each dimension
        for h in range(dimh):
            sum_sum_der_log_policy = 0
            for i in range(N):
                sum_der_log_policy = 0
                for t in range(T):
                    u = U[:,i*T+t]
                    x = X[:,i*(T+1)+t]
                    sum_der_log_policy += self.policy.calc_log_der(u,x,t)[h]
                baseline[h] += (sum_der_log_policy**2)*R[i]
                sum_sum_der_log_policy += sum_der_log_policy**2
            baseline[h] /= sum_sum_der_log_policy
        
        return baseline
        
    
    def estimate_gradient(self, D):
        '''
        Policy gradient step. Estimates the gradient.
        
        Estimates the gradient by:
        
        1. Estimating the optimal baseline to subtract from rewards.
        
        2. Estimating the derivative for each dimension.
        
        N is the number of rollouts.
        T is the horizon length.
        D is the dictionary of state-action sequences and attached rewards.
        '''
        
        N = self.rolloutsize
        T = self.horizon
        # estimate the baseline
        dimh = self.policy.dim
        R = D['rewards']
        U = D['actions']
        X = D['states']
        
        derJ = np.zeros(dimh)
        baseline = self.estimate_baseline(D)
                
        # estimate derivative of policy for each dimension
        for h in range(dimh):
            for i in range(N): # for each rollout sample
                sum_der_log_policy = 0
                for t in range(T): 
                    # calculate derivative of log likelihood of action
                    u = U[:,i*T+t]
                    x = X[:,i*(T+1)+t]
                    sum_der_log_policy += self.policy.calc_log_der(u,x,t)[h]
                derJ[h] += sum_der_log_policy * (R[i] - baseline[h]) / N
            
        return derJ
    
    def line_search(self, grad):
        '''
        Typical line search procedure for policy gradient based methods.
        
        Checking Wolfe conditions.
        
        '''
        
        #TODO: implement Wolfe condition checking!
        
        alpha = 0.0001
        
        return alpha
        
    
def test_reinforce():
    '''
    Testing the REINFORCE algorithm using a linear model class.
    '''
    
    dimx = 2
    dimy = 1
    dimu = 1
    # dimensions of the linear system, state, observation and control resp.
    dims = {'x': dimx, 'y': dimy, 'u': dimu}
    # noise covariance, process and measurement (observation) respectively
    eps = {'observation': 0.1*np.eye(dimy), 'process': 0.0*np.eye(dimx)}  
    # model matrices, A, B and C (no D given)
    A = np.array([[0, 1], \
                  [1.8, -0.81]])
    B = np.array([[0], [1]])
    C = np.array([0, 1])
    models = {'A': A, 'B': B, 'C': C}    
    # initialize the linear model class
    lin = Linear(dims, eps, models)
    
    # create a policy
    theta = np.array([0.1])
    var_policy = 0.1*np.eye(dimu)
    features = lambda x: x
    policy = LinearPolicy(1,features,var_policy,theta)
    xdes = 1
    reward = lambda x,u: -(x[:,-1] - xdes)**2
    
    rl = Reinforce(policy, reward, 10, 5)
    x0 = np.zeros(dimx)
    # learn a policy and give verbose output
    rl.learn(1e-3, x0, lin, True) 
    
print('Testing REINFORCE on a simple 1d problem with horizon = 10, rollouts = 5...')    
test_reinforce()