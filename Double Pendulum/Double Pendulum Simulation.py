#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from scipy.integrate import odeint
import timeit
import sys

class DoublePendulum(object):    
    """
    Model the properties and dynamics of a double pendulum with a rigid (but massless) rod of length l
    and a bob with mass m.
    """

    # class attributes
    g = 9.8  # acceleration due to gravity, in m/s^2
    l1 = 1.0  # length of pendulum 1 in m
    m1 = 1.0  # mass of pendulum 1 in kg
    l2 = 1.0  # length of pendulum 2 in m
    m2 = 1.0  # mass of pendulum 2 in kg

    def __init__(self,
                 th1=1.0,
                 w1=0.0,
                 l1=None,
                 m1=None,
                 th2=1.0,
                 w2=0.0,
                 l2=None,
                 m2=None):
        """
        Initialize the pendulum
        
        Args:
        th1           : starting angle from vertical (degrees)
        w1            : starting angular velocity (degrees per second)
        l1 (optional) : length of pendulum rod
        m1 (optional) : mass of pendulum bob  
        th2           : starting angle from vertical (degrees)
        w2            : starting angular velocity (degrees per second)
        l2 (optional) : length of pendulum rod
        m2 (optional) : mass of pendulum bob 
        """
        
        # initial state, packed into a single 1D array
        # internally, always use radians not degrees
        self.state = np.radians([th1, th2, w1, w2])
        
        # is these are input, override the class variables.
        if l1 is not None: self.l1 = l1
        if m1 is not None: self.m1 = m1
        if l2 is not None: self.l2 = l2
        if m2 is not None: self.m2 = m2
        
        # calculate the period in the small angle limit using formula from 1st year Physics
        self.period_0 = 2*np.pi*np.sqrt(self.l1/self.g)
        
    def a(self):
        '''
        Calculate the acceleration acting on the pendulum in its current state due to gravity
        Return the acceleration. If self is an array then give special treatment (for ODEInt)
        '''
        
        m1 = self.m1
        l1 = self.l1
        m2 = self.m2
        l2 = self.l2
        g = self.g
        
        if isinstance(self, np.ndarray):
            theta1 = self[0]
            omega1 = self[2]
            theta2 = self[1]
            omega2 = self[3]
            
        else:
            theta1 = self.state[0]
            theta2 = self.state[1]
            omega1 = self.state[2]
            omega2 = self.state[3]
            
        theta_a1 = (-g*(2*m1+m2)*sin(theta1) - m2*g*sin(theta1-2*theta2)            - 2*sin(theta1-theta2)*m2*((omega2**2)*l2+(omega1**2)*l1*cos(theta1-theta2)))            /(l1*(2*m1+m2-m2*cos(2*theta1-2*theta2)))
            
        theta_a2 = (2*sin(theta1-theta2)*((omega1**2)*l1*(m1+m2)                    + g*(m1+m2)*cos(theta1) + (omega2**2)*l2*m2*cos(theta1-theta2)))                    /(l2*(2*m1+m2-m2*cos(2*theta1-2*theta2)))
            
        return theta_a1, theta_a2
        
    def deriavtive(self, t=None):
        '''
        This takes the current state provided and calculates the derivative of the phase
        The output is an array with angular speed and angular acceleration.
        If self is an array then give special treatment (for ODEInt)
        '''
        
        theta_a1, theta_a2 = DoublePendulum.a(self)
        if isinstance(self, np.ndarray):
            dpdt = np.array([self[2], self[3], theta_a1, theta_a2])
        else:
            dpdt = np.array([self.state[2], self.state[3], theta_a1, theta_a2])
        return dpdt

    def integrate(self, t, method):
        '''
        integrate the pendulum over a given time period
        
        Args:
        t      : array of times to integrate over
        method : the method to be used for integration (Euler or Leapfrog)
        '''
        
        output = np.zeros((len(t), len(self.state)))
        output[0, :] = self.state

        if method == 'Euler':
            t0 = t[0]
            for i in range(1, len(t)):
                dt = t[i] - t0
                t0 = t[i]
                
                dpdt = self.deriavtive()
                new_state = self.state + dt*dpdt

                output [i, :] = new_state
                self.state = new_state
                
        elif method == 'Leapfrog':
            dt = t[1] - t[0] 
            
            theta_a1, theta_a2 = self.a()
            v_half_h1 = self.state[2] + dt/2*theta_a1
            v_half_h2 = self.state[3] + dt/2*theta_a2
            
            t0 = t[0]
            for i in range(1, len(t)):
                dt = t[i] - t0
                t0 = t[i]
                
                new_state = np.array([self.state[0] + dt*v_half_h1, self.state[1] + dt*v_half_h2,                                      v_half_h1, v_half_h2])
                self.state = new_state
                
                theta_a1, theta_a2 = self.a()
                v_half_h1 = v_half_h1 + dt*theta_a1
                v_half_h2 = v_half_h2 + dt*theta_a2
                output[i, :] = self.state 
                
        elif method == 'ODEInt':
            output = odeint(SimplePendulum.deriavtive, self.state, t)
                
        return output


# In[2]:


def make_times(t_stop, dt=0.01, reverse=False):
    """
    Create a array of times from 0..t_stop, sampled at dt second steps
    
    Args:
    t_stop  : ending time (not included in series)
    dt      : time step
    reverse : appends a time reversed series at the end
    
    Returns:
       array of times
    """
    
    t = np.arange(0, t_stop, dt)

    # option to reverse time and run teh simulation backlwards to the beginning
    if reverse:
            t2 = np.arange(t_stop, 0, -dt)
            t = np.concatenate((t, t2))
            
    return t


# In[3]:


def initialize_and_integrate(t, th_init1=1., w_init1=0., th_init2=1., w_init2=0., method='Euler', pendulum='Double'):
    """
    Convenience function to initialize the pendulum and integrate it

    Args:
    th_init1                : the initial angle (degrees)
    w_init1                 : the initial angular velocity (degrees per second)
    th_init2                : the initial angle (degrees)
    w_init2                 : the initial angular velocity (degrees per second)
    steps_per_period (int)  : numer of timesteps per natrual period 
    method (str)            : integration method; one of 'Euler', 'Leapfrog', 'ODEInt'
    pendulum                : the type of pendulum that is being simulated
    
    returns:
        output - phase space output, an array of angular positions and velocites at each time step
    """

    # initialize the pendulum
    if pendulum == 'Single': pend = SimplePendulum(th1=th_init1, w1=w_init1)
    elif pendulum == 'Double': pend = DoublePendulum(th1=th_init1, w1=w_init1, th2=th_init2, w2=w_init2)
    elif pendulum == 'Magnetic': pend = MagneticPendulum(x=th_init1, y=w_init1, x_dot=th_init2, y_dot=w_init2)

    # integrating pendulum
    output = pend.integrate(t, method=method)

    # separating output into angles and cartesian coordinates of each ball
    if len(output[0]) % 2 != 0:
        raise Exception('output does not have even number of variables.')
        
    return output

        
def animate(output_list, t, animation='Both', pendulum='Double', energy=False, cmap=plt.get_cmap("jet")):
    '''
    Animate the pendulum given its output and create a gif
    
    Args:
    output    :list of the phase spaces for each timestep.
    t         :the list of times used to integrate the pendulum over.
    animation :the animation style of the gif.
    pendulum  :the style of the pendulum (single, double, or magnetic).
    '''
    
    output_dict = {}
    for i in range(len(output_list)):
        output = output_list[i
                            ]
        # Separating and converting variables from output
        rads1 = output[:,0] 
        omega1 = output[:,2]
        x1 = np.sin(rads1)
        y1 = -1*np.cos(rads1)

        if pendulum == 'Double':
            rads2 = output[:,1]
            omega2 = output[:,3]
            x2 = x1 + np.sin(rads2)
            y2 = y1 - np.cos(rads2)

        if energy:
            plot_energy(pend, t, rads1, rads2, omega1, omega2, y1, y2, x1_dot, x2_dot, y1_dot, y2_dot)
            
        # establishing dictionary of all outputs and corresponding variables
        output_dict[f'output_{i}'] = {}
        output_dict[f'output_{i}'][f'output'] = output
        output_dict[f'output_{i}'][f'rads1'] = rads1
        output_dict[f'output_{i}'][f'rads2'] = rads2
        output_dict[f'output_{i}'][f'omega1'] = omega1
        output_dict[f'output_{i}'][f'omega2'] = omega2
        output_dict[f'output_{i}'][f'x1'] = x1
        output_dict[f'output_{i}'][f'y1'] = y1
        output_dict[f'output_{i}'][f'x2'] = x2 
        output_dict[f'output_{i}'][f'y2'] = y2
    N = len(output_dict)

    meta_data = {'title': 'Movie',
                'artist': 'Patrick'}
    
    # plotting 
    fig = plt.figure(figsize=(9,9))
    if pendulum == 'Magnetic':
        plt.title('Magnetic Pendulum Animation')
    elif pendulum == 'Double':
        plt.title('Double Pendulum Animation')
    plt.xlim(-4.2,4.2)
    plt.ylim(-4.2,4.2)

    if animation == 'Real Time':
        # animating pendulum
        writer = PillowWriter(fps=30, metadata=meta_data)
        with writer.saving(fig, 'TEST_PENDULUM_GIF.gif', 100):
            for i in range(len(output)):

                #plotting balls and sticks for each frame
                balls, = plt.plot([x1[i], x2[i]], [y1[i], y2[i]], 'ro')
                sticks, = plt.plot([0, x1[i], x2[i]], [0,y1[i], y2[i]], '-r')

                # taking frame then clearing plot
                writer.grab_frame()
                balls.remove()
                sticks.remove()

    elif animation == 'Both':
        for i in range(len(output_dict)):
            output_dict[f'output_{i}'][f'l'], = plt.plot([], [], '--r')
            output_dict[f'output_{i}'][f'gif_x'] = []
            output_dict[f'output_{i}'][f'gif_y'] = []

        # animating pendulum
        writer = PillowWriter(fps=30, metadata=meta_data)
        with writer.saving(fig, 'Euler Test.gif', 100):   
            print(f'Simulation {0/len(output)} complete', end='')
            for j in range(len(output)):
                for i in range(len(output_dict)):
                    output = output_dict[f'output_{i}']['output']
                    l = output_dict[f'output_{i}']['l']
                    gif_x = output_dict[f'output_{i}']['gif_x']
                    gif_y = output_dict[f'output_{i}']['gif_y']
                    x1 = output_dict[f'output_{i}']['x1']
                    y1 = output_dict[f'output_{i}']['y1']
                    x2 = output_dict[f'output_{i}']['x2']
                    y2 = output_dict[f'output_{i}']['y2']
                    
                    gif_x.append(x2[j])
                    gif_y.append(y2[j])
                    l.set_data(gif_x, gif_y)

                    #plotting balls and sticks for each frame
                    color = cmap(i/N)
                    #balls, = plt.plot([x1[j], x2[j]], [y1[j], y2[j]], marker = 'o', markerfacecolor=color)
                    sticks, = plt.plot([0, x1[j], x2[j]], [0,y1[j], y2[j]], color=color)

                # taking frame then clearing plot
                writer.grab_frame()
                plt.cla()
                plt.xlim(-4.2,4.2)
                plt.ylim(-4.2,4.2)
                
                print(f'\rSimulation {j/len(output)*100:.1f}% complete', end='', flush=True)       

    elif animation == 'Real Time':
        for i in range(len(output_dict)):
            output_dict[f'output_{i}'][f'l'], = plt.plot([], [])
            output_dict[f'output_{i}'][f'gif_x'] = []
            output_dict[f'output_{i}'][f'gif_y'] = []

        # animating pendulum
        writer = PillowWriter(fps=30, metadata=meta_data)
        with writer.saving(fig, 'Leapfrog Test.gif', 100):   
            print(f'Simulation {0/len(output)} complete', end='')
            for j in range(len(output)):
                for i in range(len(output_dict)):
                    output = output_dict[f'output_{i}']['output']
                    l = output_dict[f'output_{i}']['l']
                    gif_x = output_dict[f'output_{i}']['gif_x']
                    gif_y = output_dict[f'output_{i}']['gif_y']
                    x1 = output_dict[f'output_{i}']['x1']
                    y1 = output_dict[f'output_{i}']['y1']
                    x2 = output_dict[f'output_{i}']['x2']
                    y2 = output_dict[f'output_{i}']['y2']
                    
                    gif_x.append(x2[j])
                    gif_y.append(y2[j])
                    l.set_data(gif_x, gif_y)

                    #plotting balls and sticks for each frame
                    color = cmap(i/N)
                    #balls, = plt.plot([x1[j], x2[j]], [y1[j], y2[j]], marker = 'o', markerfacecolor=color)
                    sticks, = plt.plot([0, x1[j], x2[j]], [0,y1[j], y2[j]], color=color)

                # taking frame then clearing plot
                writer.grab_frame()
                plt.cla()
                plt.xlim(-4.2,4.2)
                plt.ylim(-4.2,4.2)
                
                print(f'\rSimulation {j/len(output)*100:.1f}% complete', end='', flush=True)           
                
def plot_energy(pend, t, rads1, rads2, omega1, omega2, y1, y2, x1_dot, x2_dot, y1_dot, y2_dot):
    '''
    Plot the energy of the pendulum over time t
    
    Args:
    pend   : list of the phase spaces for each timestep.
    t      : the list of times used to integrate the pendulum over.
    rads1  : angular position of the first bob
    rads2  : angular position of the second bob
    omega1 : angular velocity of the first bob
    omega2 : angular velocity of the second bob
    y1     : y position of the first bob
    y2     : y position of the second bob
    x1_dot : x velocity of the first bob
    y1_dot : y velocity of the first bob
    x2_dot : x velocity of the second bob
    y2_dot : y velocity of the second bob
    '''
    
    # calculate energy throughout simulation, simplified for m1=m2=1 and l1=l2=1
    I1 = 1/3*pend.m1*(pend.l1**2)
    I2 = 1/3*pend.m2*(pend.l2**2)

    x1_dot = cos(rads1)
    y1_dot = sin(rads2)
    x2_dot = omega1*cos(rads1) + omega2*cos(rads2)
    y2_dot = -omega1*sin(rads1) - omega2*sin(rads2)

    kinetic_energy = 0.5*(x1_dot**2 + y1_dot**2 + x2_dot**2 + y2_dot**2) + 0.5*(I1 + I2)*(omega1**2 + omega2**2)
    potential_energy = pend.g*(y1 + y2)
    total_energy = kinetic_energy + potential_energy

    initial_energy = total_energy[0]
    final_energy = total_energy[-1]
    energy_ratio = initial_energy/final_energy
    print(f'Initial energy: {initial_energy}')
    print(f'Final energy: {final_energy}')
    print(f'Energy ratio: {energy_ratio}')

    if initial_energy == 0:
        print(potential_energy[0])
        print(kinetic_energy[0])

    plt.plot(t, total_energy)
    plt.title('Energy Plot')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.show()


# In[4]:


def run_trials(th_init1=180, w_init1=0, th_init2=130, w_init2=0., steps_per_period=100, method='Leapfrog',                animation='Both', energy=False, trials=1, angle_dispersion=0.1):
    """
    Convenience function to run the pendulum in many different positions and animate them all.

    th_init1               : the initial angle (degrees)
    w_init1                : the initial angular velocity (degrees per second)
    th_init2               : the initial angle (degrees)
    w_init2                : the initial angular velocity (degrees per second)
    steps_per_period       : steps per period of the pendulum
    method (str)           : integration method; one of 'Euler', 'Leapfrog', 'ODEInt'
    animation              : animation style of gif (Real Time, Trace, Both)
    energy                 : if true plot the energy
    trials                 : the number of pendulums to run and animate at once
    angle_dispersion       : the size of the range of angle the pendulums will simulated around th_init2
    """
    
    # timing run time
    start = timeit.default_timer()
    
    # set up example pendulum to get period of each pendulum
    pend = DoublePendulum(th1=th_init1, w1=w_init1, th2=th_init2, w2=w_init2)
    
    # set up the array of times
    dt = pend.period_0/steps_per_period
    t = make_times(8*pend.period_0+dt, dt = dt, reverse=False)
    
    th2_range = np.linspace(th_init2 + angle_dispersion/2, th_init2 - angle_dispersion/2, trials)
    
    # running each trial by changing bob 2's angle
    output_list = []
    start_integration = timeit.default_timer()
    for i in range(len(th2_range)):
        th2 = th2_range[i]
        output = initialize_and_integrate(t, th_init1=th_init1, w_init1=w_init1, th_init2=th2,                                           w_init2=w_init2, method=method, pendulum='Double')
        output_list.append(output)
    end_integration = timeit.default_timer()
    print(f'Integration run-time: {(end_integration - start_integration):.2f}')
        
    animate(output_list, t, animation=animation, pendulum='Double', energy=energy)
    stop = timeit.default_timer()
    
    print('')
    print(f'Total run-time: {(stop - start):.2f}')
    
run_trials(th_init1=80, w_init1=0, th_init2=180, w_init2=0, steps_per_period=100, method='Leapfrog',            animation='Real Time', energy=False, trials=1, angle_dispersion=0)


# In[12]:




