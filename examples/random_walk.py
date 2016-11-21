'''
Created on Nov 15, 2016

@author: okoc
'''

import numpy as np
import pylab as plt

def random_walk_sim():
    ''' Simulates many random walks and looks at statistics'''
    
    n_stories = 1000
    t_max = 200
    
    t = np.arange(t_max)
    steps = 2 * np.random.random_integers(0,1,(n_stories,t_max)) - 1
    positions = np.cumsum(steps, axis = 1)
    sq_distance = positions**2
    mean_sq_distance = np.mean(sq_distance, axis = 0)
    
    #plot the results and the stats
    plt.figure(figsize = (4,3))
    plt.plot(t, np.sqrt(mean_sq_distance), 'b.', t, np.sqrt(t), 'r-')
    plt.ylabel(r"$\sqrt{\langle (\delta x)^2 \rangle}$")