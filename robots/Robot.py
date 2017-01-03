u'''
Created on Nov 17, 2016

Includes abstract robot class which
robot lower classes will inherit from.
Mostly as an interface.

@author: okoc
'''

class Robot(object):
    '''
    Robot is an abstract class
    which individual robots will inherit from.
    '''

    def __init__(self, params):
        '''
        Constructor
        '''
        pass
    
    def kinematics(self):
        '''
        Kinematics function for the robot.
        '''
        
    def inverse_kinematics(self):
        '''
        Inverse kinematics for the robot.
        '''
        
    def dynamics(self):
        '''
        Forward dynamics for the robot.
        Calculates accelerations given state (positions, velocities)
        and torques (control commands u).
        '''
        
    def inverse_dynamics(self):
        '''
        Inverse dynamics for the robot.
        Calculates desired torques given desired states 
        and desired accelerations (i.e. q_des, qd_des, qdd_des)
        '''
    
    def animate(self):
        '''
        Animation method for the robot
        '''
        

    def calc_racket_state(self, q, qd):
        ''' Calculate racket state
        
        '''    
        x = []
        xd = []
        o = []
        
        return x,xd,o