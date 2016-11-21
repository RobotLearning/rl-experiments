from samba import net
u'''
Created on Nov 18, 2016

This module contains the table class (more like a structure containing
values, since table doesn't really do anything). Table contains a net class!

@todo: and also functions for simulating the table. 


@author: okoc
'''

import numpy as np

class Table(object):
    u'''
    Table class containing table parameters [height, length, etc.]
    '''

    def __init__(self, params = None):
        u'''
        Constructor initializing the table. 
        
        We set the center of the table as a global origin,
        unlike the previous code elsewhere.
        '''
        
        self.height = 0.76
        self.length = 2.76
        self.width = 1.525
        # we're setting centre of table to global origin!
        self.z = 0.0 
        self.center_x = 0.0
        self.center_y = 0.0
        self.thickness = 0.02
        self.net = Net()
        self.floor_level = self.z - self.height
        
        self.set_coeffs([0.68,0.72,0.86])
        
    def set_coeffs(self, coeffs):
        
        self.CRT = -np.abs(coeffs[2])
        self.CFTX = coeffs[0]
        self.CFTY = coeffs[1]
        
class Net(object):
    u'''
    Net class belonging to a table.
    Contains net parameters [length, height, etc.]
    as well as a nominal reflection law
    '''
    
    def __init__(self, params = None):
        u'''
        Constructor initializing the net parameters.
        '''
        self.height = 0.144
        self.overhang = 0.12 # for simulation
        self.thickness = 0.01
        
        self.set_restitution_coeff(0.05)
        
    def set_restitution_coeff(self, coeff = 0.05):
        u'''
        Sets the restitution coefficient of the net.
        For ball hitting the net during simulation.
        '''
        
        assert(coeff < 0.1)
        self.restitution = coeff