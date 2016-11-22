u'''
Created on Nov 18, 2016

This module contains the table class (more like a structure containing
values, since table doesn't really do anything). Table contains a net class!

@todo: and also functions for simulating the table. 


@author: okoc
'''

import numpy as np

class Table(object):
    '''
    Table class containing table parameters [height, length, etc.]
    '''

    def __init__(self, params = None):
        '''
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
    '''
    Net class belonging to a table.
    Contains net parameters [length, height, etc.]
    as well as a nominal reflection law
    '''
    
    def __init__(self, params = None):
        '''
        Constructor initializing the net parameters.
        '''
        self.height = 0.144
        self.overhang = 0.12 # for simulation
        self.thickness = 0.01
        
        self.set_restitution_coeff(0.05)
        
    def set_restitution_coeff(self, coeff = 0.05):
        '''
        Sets the restitution coefficient of the net.
        For ball hitting the net during simulation.
        '''
        
        assert(coeff < 0.1)
        self.restitution = coeff
        
# Test functions here for simulating with mlab
def test_sim_table():
    '''
    Simulates 2D-table using 3D graphics of mlab
    '''        
    
    from mayavi import mlab

    table = Table()
    xmin = table.center_x - table.width/2.0
    xmax = table.center_x + table.width/2.0
    ymin = table.center_y - table.length/2.0
    ymax = table.center_y + table.length/2.0
    xnum = np.complex(0,2)
    ynum = np.complex(0,2)
    x, y = np.mgrid[xmin:xmax:xnum, ymin:ymax:ynum]
    z = np.zeros(x.shape)
    green = (0, 0.7, 0.3)
    s = mlab.surf(x, y, z, warp_scale = 'auto', color = green)
    
def test_sim_3d_table():
    '''
    Renders a 3d mesh using table coordinates.
    '''

    from mayavi import mlab

    table = Table()
    xmin = table.center_x - table.width/2.0
    xmax = table.center_x + table.width/2.0
    ymin = table.center_y - table.length/2.0
    ymax = table.center_y + table.length/2.0
    table.thickness = 0.10
    zmin = table.z - table.thickness
    zmax = table.z
    
    num = np.complex(0,2) # number of points to include per axis
    x,y = np.mgrid[xmin:xmax:num, ymin:ymax:num]
    xx = np.c_[x,x]
    yy = np.c_[y,y]
    z = np.c_[zmin * np.ones(x.shape),zmax * np.ones(x.shape)]
    
    green = (0, 0.7, 0.3)
    
    face_idx = [[1, 2, 3, 4], 
              [5, 6, 7, 8], 
              [1, 2, 6, 5], 
              [2, 3, 7, 6], 
              [3, 4, 8, 7], 
              [4, 1, 5, 8]]
    face = np.asarray(face_idx)
    
    for i in range(len(face_idx)):
        x_face = xx[:,face[i]-1]
        y_face = yy[:,face[i]-1]
        z_face = z[:,face[i]-1]
        mlab.mesh(x_face, y_face, z_face, color = green)    

test_sim_3d_table()