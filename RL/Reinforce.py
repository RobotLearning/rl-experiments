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
    
    def __init__(self, policy, reward, horizon, num_rollouts, alpha):
        '''
        Constructor.
        '''
        self.policy = policy
        self.reward = reward
        self.horizon = horizon
        self.rolloutsize = num_rollouts
        self.alpha = alpha
        
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
            # gradient descent
            theta_last = self.policy.theta.copy()
            self.policy.theta += self.alpha * grad
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
        
    
def test_reinforce():
    '''
    Testing the REINFORCE algorithm using a linear model class.
    '''
    
    dimx = 4
    dimy = 4
    dimu = 2
    # dimensions of the linear system, state, observation and control resp.
    dims = {'x': dimx, 'y': dimy, 'u': dimu}
    # noise covariance, process and measurement (observation) respectively
    eps = {'observation': 1e-4*np.eye(dimy), 'process': 0.0*np.eye(dimx)}
    # model matrices, A, B and C (no D given)
    
    A = np.random.rand(dimx,dimx)
    B = np.random.rand(dimx,dimu)
    C = np.eye(dimx, dimy)
    #a = 0.9 # damping constant
    #A = np.array([[0, 1], \
    #              [-a*a, -2*a]])
    #B = np.array([[0], [1]])
    #C = np.eye(dimy) #np.array([1, 0])
    
    models = {'A': A, 'B': B, 'C': C}  
    # initialize the linear model class
    lin = Linear(dims, eps, models)
    
    # create a policy
    theta = -np.random.rand(dimu,dimy) #np.array([-0.1,-0.1])
    var_policy = 0.001*np.eye(dims['u'])
    features = lambda x: x
    policy = LinearPolicy(np.size(theta),features,var_policy,theta)
    xdes = np.random.randn(dimy)
    reward = lambda x,u: -np.dot((x[:,-1] - xdes),x[:,-1] - xdes)
    
    horizon = 10
    num_rollouts = 50
    tol = 1e-6
    alpha = 1e-3
    rl = Reinforce(policy, reward, horizon, num_rollouts,alpha)
    x0 = np.zeros(dims['x'])
    # learn a policy and give verbose output
    rl.learn(tol, x0, lin, True) 
    
if __name__ == "__main__":
    print('Testing REINFORCE on a simple 1d problem...')    
    test_reinforce()