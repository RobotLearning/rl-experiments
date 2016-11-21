u'''
Created on Nov 17, 2016

@author: okoc
'''

class Robot(object):
    u'''
    classdocs
    '''


    def __init__(self, params):
        u'''
        Constructor
        '''
        pass
    
    def evolve(self, dt, torque):
        u'''
        Evolve the robot using forward dynamics        
    
        '''
        q = []
        qd = []    
    
        return q,qd
        

    def calc_racket_state(self, q, qd):
        u''' Calculate racket state
        
        '''    
        x = []
        xd = []
        o = []
        
        return x,xd,o