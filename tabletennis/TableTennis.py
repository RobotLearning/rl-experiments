u'''
Created on Nov 17, 2016

Table tennis class for making table tennis animations.

@attention: Rewritten from original MATLAB repo

@author: okoc

'''

from tabletennis.Robot import Robot
from tabletennis.Ball import Ball

class TableTennis(object):
    u'''
    Table tennis 3D class for making table tennis animations.
    
    @todo: Extend for 2 robots playing against each other!
    '''


    def __init__(self):
        u'''
        Constructor. Set sampling time (dt).
        
        Sets the score to zero: for now the score is the number of successful
        returns to the other side. 
        @todo: Can be extended for 2 robots playing!
        
        '''
        
        print('Welcome to the Table Tennis 3D simulation')
        self.score = 0
        
    def set_robot(self, robot):
        u''' 
        Set robot playing table tennis. 
        
        @todo: Extend for multiple robots.         
               Check if robot is a Robot instance.
        
        '''
        
        self.robot = robot
        
    def set_ball(self, ball):
        u'''
        Set ball for the table tennis simulation.
        
        @todo: Check if ball is instance of ball3d.
        '''
        
        self.ball = ball
        
    def set_samp_time(self, samp_time = 0.01):
        u'''
        Set sampling time (dt).
        
        Checks if sampling time is within range [0,0.1]
        
        '''
        
        samp_range = [0,0.1]
        assert(samp_time > samp_range[0] and \
               samp_time < samp_range[1] , \
               "Sampling time is not right!")
        self.dt = samp_time
        
    def set_table(self, options):
        u'''
        Set the table specifics here.
        '''
        
        self.table = options
        
    def set_vision(self, options):
        u'''
        Set the details of how we perceive the environment:
        ball state.        
        
        '''
        
        self.vision = options
        
    def practice(self, num_times = 1, max_sim_time = 3.0):
        u'''
        Practice playing table tennis with a robot and a ball machine. 
        
        The input parameter num_times indicates the number of solo trials.
        Maximum simulation time is preset to 3.0 seconds for each trial.
        
        @todo: Extend for two robots playing against each other! 
        
        '''
        
        for i in range(num_times):
            self.play_one_turn(max_sim_time)
            print('Iteration %d' % i)
            
        print('Landed %d out of %d' % self.score, num_times)    
            
    def play_one_turn(self, max_sim_time):
        u'''
        The robot will try to hit the ball (only once!).
        
        Argument indicates the maximum simulation time that will be spent
        waiting for the interaction.
        
        @todo: If ball lands make sure to increase the score by one!
        
        @todo: Assert that the robot implements the right methods
        @todo: Make sure robot goes back to q0!
        
        ''' 
        
        time_passed = 0.0
        
        while time_passed < max_sim_time:
            
            dt = self.dt
            # evolve the ball dt seconds and check interactions
            self.ball.evolve(dt, self.table, self.robot.racket)
            # get ball estimate
            est = self.get_ball_estimate()
            # plan for robot motion using ball estimate
            q,qd = self.plan(est)
            # get cartesian state for racket
            x,xd,o = self.robot.calc_racket_state(q,qd)
            # increment time passed
            time_passed += dt
            
    def plan(self, ball_estimate):
        u'''        
        Plan using a Virtual Planning Plane (VPP) set to the net y-position.
        
        Using a finite state machine to plan when to hit and to stop.        
        Input argument is an estimate of the ball
        
        @todo:  Remove the VPP and do it repeatedly.                
        
        '''
        
        VPP_MODE = True
        _Y_ = 1
        _DY_ = 4
        vel_thresh = 0.5
        
        if VPP_MODE:
            if ball_estimate[_Y_] > self.table.center and \
               ball_estimate[_DY_] > vel_thresh:
                tau = self.calc_opt_traj(ball_estimate)
        else:
            raise NotImplementedError('Not implemented yet!')
        
        q,qd = self.robot.evolve(tau)
        return q,qd
        
    def calc_opt_traj(self, est):
        u'''
        
        Calculate optimal hitting and returning trajectories
        using 3rd order polynomials.
        
        Running a separate optimization routine
        @todo: 
        
        
        '''
        tau = 0
        
        return tau # torques
            
    def get_ball_estimate(self):
        u'''
        Get ball estimate from the vision system.
        
        '''
        est = []
        
        return est
        
# set all the options         
VISION_OPTS = {}
TABLE_OPTS = {}
        
tt = TableTennis()
wam = Robot()
ball = Ball()
tt.set_robot(wam)
tt.set_ball(ball)

tt.set_vision(VISION_OPTS)
tt.set_samp_time(0.01)
tt.set_table(TABLE_OPTS)
tt.practice()