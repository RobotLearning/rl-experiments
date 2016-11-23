u'''
Created on Nov 17, 2016

@author: okoc
'''

import numpy as np
from scipy import integrate

from tabletennis.Table import Table 

class Ball(object):
    '''
    Ball class used to simulate table tennis with a robot
    '''

    def __init__(self, dt, params):
        '''
        Constructor
        
        @todo:  Check that the params dictionary contains the right
        keys
        '''
        
        #assert(dt < 0.02 and dt > 0.0, 'please provide a small time step dt!')
        self.dt = dt
        self.params = params
        
    def set_state(self, pos, vel):
        '''
        Set the initial ball state: initial ball position and velocity.
        
        '''      
        
        self.state = np.concatenate([pos,vel]) 
        
    def ball_flight_model(self, xdot):
        '''
        
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
        '''
        
        Computes the velocities and accelerations given current positions
        and velocities.
        
        '''
        
        _DX_ = 3
        pos = state[:_DX_]
        vel = state[_DX_:]
        dy = np.concatenate([vel,self.ball_flight_model(vel)])
        return dy
        
    def evolve(self, t_evolve, table, racket):
        '''
        Evolve the ball for dt seconds.
        TODO: implement Runge Kutta 4 integration!
        so far using Symplectic Euler! 
        
        Checks interaction with table, net, racket and ground.
        '''
        
        fun = self.ball_fun
        n_evolve = int(t_evolve/self.dt)
        #time = np.linspace(0,t_evolve,int(t_evolve/self.dt))
        #time = np.array([0, t_evolve])
        
        _DX_ = 3 # index of x-velocity
        next_state = self.state.copy()
        for i in range(n_evolve): #time:
            # integrating with Symplectic Euler
            next_state[_DX_:] = next_state[_DX_:] + \
            self.dt * self.ball_flight_model(self.state[_DX_:])
            # integrate the positions
            next_state[:_DX_] = next_state[:_DX_] + \
            self.dt * next_state[_DX_:]
            # check contact with table, net, racket, ground
            next_state = self.check_contact(next_state, table, racket)                    
            self.state = next_state.copy()
        
        #sol = integrate.odeint(fun, self.state, time)
        #next_state = sol[-1,:]
        
        
    def check_contact(self, next_state, table, racket):
        '''
        Checks contact with the table, racket, net and floor,
        in that order.
        
        If there is contact, then the state already evolved according
        to a flight model will be modified.
        
        '''
        
        next_state = self.check_contact_table(next_state, table)
        #TODO:
        
        return next_state
        
    def check_contact_table(self, next_state, table):
        '''
        Checks contact with table by checking if ball would cross
        the table in z-direction if it moved freely.
        
        dt here is used to predict ball positions and velocities accurately.
        More precisely, we estimate the time t_reflect \in [0,dt]
        and evolve the ball till there, reflect ball velocities and then 
        continue predicting for the next dt - t_reflect seconds.  
        '''
        
        _Z_ = 2
        _Y_ = 1
        _X_ = 0
        _DX_ = 3
        z_next = next_state[_Z_]
        y_next = next_state[_Y_]
        x_next = next_state[_X_]
        z_cur = self.state[_Z_]
        # check if ball would cross the table - in z direction
        cross = (np.sign(z_next - table.z) != (np.sign(z_cur - table.z)))
        # check if ball is over the table
        over = (np.abs(y_next - table.center_y) < table.length/2.0) and \
                (np.abs(x_next - table.center_x) < table.width/2.0)
        
        if cross and over:
            # find bounce time
            # doing bisection to determine bounce time
            tol = 1e-4
            dt1 = 0.0
            dt2 = self.dt # for bracketing
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
            dt = self.dt - dt_bounce
            # integrate for the remaining time dt -
            next_state[_DX_:] = bounce_state[_DX_:] + \
                    dt * self.ball_flight_model(bounce_state[_DX_:])
            next_state[:_DX_] = bounce_state[:_DX_] + \
                    dt * next_state[_DX_:]
                    
        return next_state    
        
    def rebound_model(self, table, xdot):
        '''
        Apply reflection law on the velocities.
        '''
        
        M = np.diag([table.CFTX, table.CFTY, table.CRT], 0)
        return M.dot(xdot)
             
        
        
# Test function here for the Ball class

def test_ball_freefall(t_evolve = 0.2):
    '''
    Testing whether ball.state returns us the right position 
    and velocity values after evolving for dt seconds


    '''
    dt = 0.01
    C = 0.1414
    g = -9.81
    params = [C,g]        
    ball = Ball(dt, params)
    table = Table()   
    init_pos = np.ones(3)
    init_vel = np.zeros(3)
    ball.set_state(init_pos, init_vel)
    ball.evolve(t_evolve, table, [])
    print('Evolving state for %f seconds' % dt)
    print('Ball pos and vel:')    
    np.set_printoptions(precision=3)
    print(ball.state)
    
def test_ball_bounce(t_evolve = 1.0):
    '''
    Testing whether ball bounces properly on the table
    
    ''' 
    
    dt = 0.01
    C = 0.1414
    g = -9.81
    params = [C,g]        
    ball = Ball(dt,params)  
    table = Table()
    init_pos = np.array([0.8197, -1.62, 0.32])
    init_vel = np.array([-2.2230, 5.7000, 2.8465])
    ball.set_state(init_pos, init_vel)
    ball.evolve(t_evolve, table, [])
    print('Evolving state for %f seconds' % t_evolve)
    print('Ball pos and vel:')
    np.set_printoptions(precision=3)
    print(ball.state)   
    
def test_ball_sim():
    '''
    Drawing the ball and animating with mlab
    '''
    
    from mayavi import mlab
    import time
    
    # to prevent error messages
    import vtk
    output = vtk.vtkFileOutputWindow()
    output.SetFileName("log.txt")
    vtk.vtkOutputWindow().SetInstance(output)
    
    p = np.array([0.8197, -1.62, 0.32])
    #init_vel = np.array([-2.2230, 5.7000, 2.8465])
    orange = (0.9100, 0.4100, 0.1700)
    s = mlab.points3d(p[0],p[1],p[2],color = orange, scale_factor = 0.05)
    for i in range(10):
        #time.sleep(0.2)
        p = p + 0.01
        s.mlab_source.set(x = p[0],y = p[1], z = p[2])
