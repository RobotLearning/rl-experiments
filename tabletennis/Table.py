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

    # get the coordinates
    table = Table()
    xmin = table.center_x - table.width/2.0
    xmax = table.center_x + table.width/2.0
    ymin = table.center_y - table.length/2.0
    ymax = table.center_y + table.length/2.0
    # table.thickness = 0.10
    zmin = table.z - table.thickness
    zmax = table.z
    
    # make the points of the rectangular grid
    # first make the indices
    z_idx = np.r_[np.zeros(4),np.ones(4)]
    y_idx = np.array([0,1,0,1])
    y_idx = np.r_[y_idx,y_idx]
    x_idx = np.array([0,0,1,1])
    x_idx = np.r_[x_idx,x_idx]
    # now make the 3d points
    pts_x = xmin + (xmax-xmin) * x_idx
    pts_y = ymin + (ymax-ymin) * y_idx
    pts_z = zmin + (zmax-zmin) * z_idx
    
    green = (0, 0.7, 0.3)
    
    # now construct the faces
    face_idx = [[1, 2, 3, 4], 
              [5, 6, 7, 8], 
              [1, 2, 5, 6], 
              [3, 1, 7, 5], 
              [3, 4, 7, 8], 
              [4, 2, 8, 6]]
    face = np.asarray(face_idx)
    # draw the faces one by one     
    for i in range(len(face_idx)):
        x_face = pts_x[face[i]-1].reshape(2,2)
        y_face = pts_y[face[i]-1].reshape(2,2)
        z_face = pts_z[face[i]-1].reshape(2,2)
        mlab.mesh(x_face, y_face, z_face, color = green)    
