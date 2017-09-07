'''
Created on Nov 15, 2016

@author: okoc
'''

import numpy as np
import matplotlib.pylab as plt

def poly_fit(output='poly.png'):
    ''' Polynomial fitting example of order 3. '''
    
    x = np.linspace(0, 1, 20)
    y = np.cos(x) + 0.3*np.random.randn(20)
    p = np.poly1d(np.polyfit(x, y, 3))
    t = np.linspace(0, 1, 200)
    plt.plot(x, y, 'o', x, np.cos(x), '-r', t, p(t), '-')
    plt.savefig(output, dpi=96)
    
def chebyshev_fit(output='chebyshev.png'):
    ''' Poly fitting with Chebyshev bases'''
    
    x = np.linspace(-1, 1, 2000)
    y = np.cos(x) + 0.3*np.random.randn(2000)
    p = np.polynomial.Chebyshev.fit(x, y, 90)
    t = np.linspace(-1, 1, 200)
    plt.clf()
    plt.plot(x, y, 'r.')
    plt.plot(t, p(t), 'k-', lw=3)    
    plt.savefig(output, dpi=96)
    
print('Testing polyfitting...')
poly_fit()
print('Testing chebyshev fitting...')
chebyshev_fit()
        