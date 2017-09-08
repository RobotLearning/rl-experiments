'''
Created on Nov 15, 2016

@author: okoc
'''

import numpy as np


def rough_prime_sieve(n):
    '''Prime sieve of Erathosthenes.
    
    Works by marking iteratively multiples of increasing p > 1.
    Refinements to speed up algorithm are possible.
    '''
    
    is_prime = np.ones((n,), dtype = bool)
    is_prime[:2] = 0 
    N_max = int(np.sqrt(n))
    
    # here is the main part
    for j in range(2,N_max):
        is_prime[2*j::j] = False
        
    print(np.nonzero(is_prime))
    
def optim_prime_sieve(n):
    ''' Slightly optimized prime sieve, see rough_prime_sieve'''
    
    is_prime = np.ones((n,), dtype = bool)
    is_prime[:2] = 0 
    # manually eliminating 2p numbers
    is_prime[2] = 1
    is_prime[4::2] = False
    N_max = int(np.sqrt(n))
    
    # here is the main part
    odds = range(3,N_max+1,2)
    for j in odds:
        is_prime[2*j::j] = False
        
    print(np.nonzero(is_prime))