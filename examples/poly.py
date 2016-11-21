'''
Created on Nov 15, 2016

@author: okoc
'''

import numpy as np
import matplotlib.pylab as plt

def poly_fit():
    ''' Polynomial fitting example of order 3. '''
    
    x = np.linspace(0, 1, 20)
    y = np.cos(x) + 0.3*np.random.rand(20)
    p = np.poly1d(np.polyfit(x, y, 3))
    t = np.linspace(0, 1, 200)
    plt.plot(x, y, 'o', t, p(t), '-')
    
def chebyshev_fit():
    ''' Poly fitting with Chebyshev bases'''
    
    x = np.linspace(-1, 1, 2000)
    y = np.cos(x) + 0.3*np.random.rand(2000)
    p = np.polynomial.Chebyshev.fit(x, y, 90)
    t = np.linspace(-1, 1, 200)
    plt.plot(x, y, 'r.')
    plt.plot(t, p(t), 'k-', lw=3)    