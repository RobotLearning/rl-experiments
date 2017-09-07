'''
Created on Nov 16, 2016

@author: okoc
'''

import numpy as np

def check_mc_conv():
    ''' Check markov chain eigenvector convergence with an example'''
    
    num_states = 5
    num_iter = 500  
    
    M = np.random.rand(num_states,num_states)    
    p = np.random.rand(num_states,1)
    
    # normalize matrix and initial distribution p
    M = M / np.sum(M, axis = 1)
    p = p / np.sum(p)
    
    #assert(all(np.sum(M, axis = 1) == 1))
    #assert(p.sum() == 1) 
    
    # compute eigenvector of M transpose
    l, V = np.linalg.eig(M.transpose())
    p_eig = V[:,l.argmax()]
    p_stat = p_eig / np.sum(p_eig)
    
    for i in range(num_iter):
        p = M.transpose().dot(p)
        
    return p_stat, p
    

print('Calling Markov chain convergence function...')
p_stat, p = check_mc_conv()        
print('Stationary eigenvector:', p_stat)
print('Approximation:', p)
    
    