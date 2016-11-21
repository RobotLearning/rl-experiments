u'''
Created on Nov 17, 2016

@author: okoc
'''

import numpy as np
from scipy import integrate

from tabletennis.Table import Table 

class Ball(object):
    u'''
    Ball class used to simulate table tennis with a robot
    '''

    def __init__(self, params):
        u'''
        Constructor
        
        @todo:  Check that the params dictionary contains the right
        keys
        '''
        self.params = params
        
    def set_state(self, pos, vel):
        u'''
        Set the initial ball state: initial ball position and velocity.
        
        '''      
        
        self.state = np.concatenate([pos,vel]) 
        
    def ball_flight_model(self, xdot):
        u'''
        
        Computes the accelerations given the current position and velocity
        of the ball. 
        '''
        
        C, g = self.params
        v = np.sqrt(xdot[0]**2 + xdot[1]**2 + xdot[2]**2)
        
        xddot = np.zeros(3)
        
        xddot[0] = -C*v*xdot[0]
        xddot[1] = -C*v*xdot[1]
        xddot[2] = g - C*v*xdot[2]
        
        return xddot
    
    def ball_fun(self,state,time):
        u'''
        
        Computes the velocities and accelerations given current positions
        and velocities.
        
        '''
        
        _DX_ = 3
        pos = state[:_DX_]
        vel = state[_DX_:]
        dy = np.concatenate([vel,self.ball_flight_model(vel)])
        return dy
        
    def evolve(self, dt, table, racket):
        u'''
        Evolve the ball for dt seconds. 
        
        Checks interaction with table, net, racket and ground.
        '''
        
        fun = self.ball_fun
        time = np.array([0, dt])
        sol = integrate.odeint(fun, self.state, time)    
        next_state = sol[-1,:]
        
        # check contact with table, net, racket, ground
        next_state = self.check_contact(dt, next_state, table, racket)
        
        self.state = next_state
        
    def check_contact(self, dt, next_state, table, racket):
        u'''
        Checks contact with the table, racket, net and floor,
        in that order.
        
        If there is contact, then the state already evolved according
        to a flight model will be modified.
        
        '''
        
        next_state = self.check_contact_table(dt, next_state, table)
        #TODO:
        
        return next_state
        
    def check_contact_table(self, dt, next_state, table):
        u'''
        Checks contact with table by checking if ball would cross
        the table in z-direction if it moved freely.
        
        dt here is used to predict ball positions and velocities accurately.
        More precisely, we estimate the time t_reflect \in [0,dt]
        and evolve the ball till there, reflect ball velocities and then 
        continue predicting for the next dt - t_reflect seconds.  
        '''
        
        _Z_ = 2
        _Y_ = 1
        _DX_ = 3
        z_next = next_state[_Z_]
        y_next = next_state[_Y_]
        z_cur = self.state[_Z_]
        # check if ball would cross the table - in z direction
        cross = (np.sign(z_next - table.z) is not (np.sign(z_cur - table.z)))
        # check if ball is over the table
        over = (np.abs(y_next - table.center_y) < table.length/2.0)
        
        if cross and over:
            # find bounce time
            # doing bisection to determine bounce time
            tol = 1e-4
            dt1 = 0.0
            dt2 = dt # for bracketing
            bounce_state = self.state.copy()
            dt_bounce = 0.0
            while np.abs(bounce_state[_Z_] - table.z) > tol:
                dt_bounce = (dt1 + dt2)/2.0
                # Symplectic Euler integration
                bounce_state[_DX_:] = self.state[_DX_:] + \
                        dt_bounce * self.ball_flight_model(self.state[_DX_:])
                bounce_state[:_DX_] = self.state[:_DX_] + \
                        dt_bounce * bounce_state[_DX_:]
                # if different signs 
                if np.sign(z_next - table.z)*(bounce_state[_Z_] - table.z) < 0:
                    # increase the expected bounce time
                    dt1 = dt_bounce
                else:
                    dt2 = dt_bounce
                    
            # apply rebound model to velocities
            bounce_state[_DX_:] = self.rebound_model(table, bounce_state[_DX_:]) 
            dt = dt - dt_bounce
            # integrate for the remaining time dt -
            next_state[_DX_:] = bounce_state[_DX_:] + \
                    dt * self.ball_flight_model(bounce_state[_DX_:])
            next_state[:_DX_] = bounce_state[:_DX_] + \
                    dt * next_state[_DX_:]
                    
        return next_state    
        
    def rebound_model(self, table, xdot):
        u'''
        Apply reflection law on the velocities.
        '''
        
        M = np.diag([table.CFTX, table.CFTY, table.CRT], 0)
        return M.dot(xdot)
             
        
        
# testing Ball class        

def test_ball_freefall(dt = 0.2):
    u'''
    Testing whether ball.state returns us the right position 
    and velocity values after evolving for dt seconds
    
    
    '''
    C = 0.1414
    g = -9.81
    params = [C,g]        
    ball = Ball(params)   
    init_pos = np.ones(3)
    init_vel = np.zeros(3)
    ball.set_state(init_pos, init_vel)
    ball.evolve(dt, [], [])
    print('Evolving state for %f seconds' % dt)
    print('Ball pos and vel:')
    print(ball.state)     
    
def test_ball_bounce(t_evolve = 1.0):
    u'''
    Testing whether ball bounces properly on the table
    
    '''    
    from tabletennis.Table import Table
    
    C = 0.1414
    g = -9.81
    params = [C,g]        
    ball = Ball(params)  
    table = Table()
    init_pos = np.array([0.8197, -1.62, 0.32])
    init_vel = np.array([-2.2230, 5.7000, 2.8465])
    ball.set_state(init_pos, init_vel)
    ball.evolve(t_evolve, table, [])
    print('Evolving state for %f seconds' % t_evolve)
    print('Ball pos and vel:')
    print(ball.state)   