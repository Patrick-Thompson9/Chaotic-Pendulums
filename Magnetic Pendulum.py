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

class MagneticPendulum(object):    
    """
    Model the properties and dynamics of a magnetic pendulum with a rigid (but massless) rod of length l
    and a bob with mass m.
    """

    # class attributes
    g = 9.8  # acceleration due to gravity, in m/s^2
    l = 1.0  # length of pendulum 1 in m
    m = 1.0  # mass of pendulum 1 in kg
    
    mu = 1.257e-6
    magnets = [(0.5, 0), (-0.3, -0.5)] # positions of magnets in [x, y] coords with th in degrees
    magnet_strength = 12000 # strength of the magnets
    k_f = 0.5 # friction coefficient of pendulum in kg/s
    min_height = 0.5 # height of bob above magnets when hanging limp

    def __init__(self,
                 x=1.0,
                 y=1.0,
                 x_dot=0.0,
                 y_dot=0.0,
                 l=None,
                 m=None,
                 magnets=None,
                 min_height=None):
        """
        Initialize the pendulum
        
        Args:
        x                     : starting x position
        y                     : starting y position
        x_dot                 : starting x velocity
        y_dot                 : starting y velocity
        l (optional)          : length of pendulum rod
        m (optional)          : mass of pendulum bob   
        magents               : a list of static magnet positions surrounding the pendulum
        min_height (optional) : minimum height of the pendulum's bob from the ground (magnets) when hanging limp
        """
        
        # initial state, packed into a single 1D array
        self.state = np.array([x, y, x_dot, y_dot])
        
        # is these are input, override the class variables.
        if l is not None: self.l = l
        if m is not None: self.m = m
        if magnets is not None: self.magnets = magnets
        if min_height is not None: self.min_height = min_height
    
        # calculate the period in the small angle limit using formula from 1st year Physics
        self.period_0 = 2*np.pi*np.sqrt(self.l/self.g)
        
    def a(self):
        '''
        Calculate the acceleration acting on the pendulum in its current state due to gravity, friction, and magnets.
        Return the acceleration. If self is an array then give special treatment (for ODEInt)
        '''
        
        m = self.m
        l = self.l
        g = self.g
        
        if isinstance(self, np.ndarray):
            x = self[0]
            y = self[1]
            x_dot = self[2]
            y_dot = self[3]
            
        else:
            x = self.state[0]
            y = self.state[1]
            x_dot = self.state[2]
            y_dot = self.state[3]
            
        ax_g = -self.g/self.l*x/self.m
        ay_g = -self.g/self.l*y/self.m
        
        ax_f = -self.k_f*x_dot/self.m
        ay_f = -self.k_f*y_dot/self.m
        
        ax_m = 0
        ay_m = 0
        
        for magnet in self.magnets:
            r = ((x - magnet[0])**2 + (y - magnet[1])**2)**0.5
            distance = (r**2 + self.min_height**2)**0.5
            
            ax_m -= self.mu*(self.magnet_strength**2)*(x - magnet[0])/(distance**3)
            ay_m -= self.mu*(self.magnet_strength**2)*(y - magnet[1])/(distance**3)
            
        ax_tot = ax_g + ax_f + ax_m
        ay_tot = ay_g + ay_f + ay_m
            
        return ax_tot, ay_tot

        
    def deriavtive(self, t=None):
        '''
        This takes the current state provided and calculates the derivative of the phase
        The output is an array with angular speed and angular acceleration.
        If self is an array then give special treatment (for ODEInt)
        '''
        
        ax_tot, ay_tot  = MagneticPendulum.a(self)
        if isinstance(self, np.ndarray):
            dpdt = np.array([self[2], self[3], ax_tot, ay_tot])
        else:
            dpdt = np.array([self.state[2], self.state[3], ax_tot, ay_tot])
        return dpdt

    def integrate(self, t, method):
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
            
            ax_tot, ay_tot = self.a()
            vx_half_h = self.state[2] + dt/2*ax_tot
            vy_half_h = self.state[3] + dt/2*ay_tot
            
            t0 = t[0]
            for i in range(1, len(t)):
                dt = t[i] - t0
                t0 = t[i]
                
                new_state = np.array([self.state[0] + dt*vx_half_h, self.state[1] + dt*vy_half_h,                                       vx_half_h, vy_half_h])
                self.state = new_state
                
                ax_tot, ay_tot = self.a()
                vx_half_h = vx_half_h + dt*ax_tot
                vy_half_h = vy_half_h + dt*ay_tot
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


# In[22]:


def integrate_and_animate(x=1., y=0., x_dot=1., y_dot=0., steps_per_period=100, magnets=[(0.5, 0)],                           method='Euler', animation='Real Time', plot_size=4.2, energy=False, cmap=plt.get_cmap("jet")):
    """
    Convenience function to integrate the pendulum and animate it

    x                       : starting x position
    y                       : starting y position
    x_dot                   : starting x velocity
    y_dot                   : starting y velocity
    magents                 : a list of static magnet positions surrounding the pendulum
    steps_per_period (int)  : numer of timesteps per natrual period 
    method (str)            : integration method; one of 'Euler', 'Leapfrog', 'ODEInt'
    animation               : animation style of gif (Real Time, Trace, Both)
    
    """

    # initialize the pendulum
    pend = MagneticPendulum(x=x, y=y, x_dot=x_dot, y_dot=y_dot, magnets=magnets)

    # set up the array of times
    dt = pend.period_0/steps_per_period
    t = make_times(7*pend.period_0+dt, dt = dt, reverse=False)

    # integrating pendulum
    start_integration = timeit.default_timer() 
    output = pend.integrate(t, method=method)
    end_integration = timeit.default_timer() 
    #print(f'Integration run-time: {(end_integration - start_integration):.2f}')

    # determine which magnet it is over
    x_f = output[-1][0]
    y_f = output[-1][1]
    for i in range(len(pend.magnets)):
        if ((x_f - pend.magnets[i][0])**2 + (y_f - pend.magnets[i][1])**2) < 0.36:
            winning_magnet = i
            break
    #print(f'The bob ends over magnet {i+1}.')
      
    if animation is not None:
        animate(output=output, t=t, pend=pend, animation=animation, energy=energy, plot_size=plot_size, cmap=cmap)
        
    return winning_magnet

def animate(output, t, pend, animation='Real Time', energy=False, plot_size=4.2, cmap=plt.get_cmap("jet")):
    """
    Convenience function to animate the pendulum

    animation        : animation style of gif (Real Time, Trace, Both)
    output           : output of pendulum to animated
    t                : time array of the pendulum
    energy           : if true plot the energy
    plot_size        : size of the plot (both axes)
    cmap             : color scheme of the objects and plot
    """
    
    # plotting 
    fig = plt.figure(figsize=(9,9))
    plt.title('Magnetic Pendulum Animation')
    plt.xlim(-plot_size, plot_size)
    plt.ylim(-plot_size, plot_size)
    
    x = output[:,0]
    y = output[:,1]
    x_dot = output[:,2]
    y_dot = output[:,3]


    meta_data = {'title': 'Movie',
                'artist': 'Patrick'}

    if animation == 'Real Time':
        # animating pendulum
        writer = PillowWriter(fps=30, metadata=meta_data)
        with writer.saving(fig, 'Apr22 triangle magnet.gif', 100):
            num_of_magnets = len(pend.magnets)
            for i in range(len(pend.magnets)):
                color = cmap(i/num_of_magnets)
                plt.scatter([pend.magnets[i][0]], [pend.magnets[i][1]], s=[150], color=color)
                
            print(f'Animation {0/len(output)*100}% complete', end='')   
            for i in range(len(output)):

                #plotting balls and sticks for each frame
                balls, = plt.plot(x[i], y[i], 'ro')
                sticks, = plt.plot([0, x[i]], [0, y[i]], '-r')

                # taking frame then clearing plot
                writer.grab_frame()
                balls.remove()
                sticks.remove()
                
                print(f'\rAnimation {i/len(output)*100:.1f}% complete', end='', flush=True)

    elif animation == 'Trace':
        l, = plt.plot([], [], '--r')
        gif_x = []
        gif_y = []
        
        # animating pendulum
        writer = PillowWriter(fps=30, metadata=meta_data)
        with writer.saving(fig, 'Apr22 7 magnets.gif', 100):
            num_of_magnets = len(pend.magnets)
            for i in range(len(pend.magnets)):
                color = cmap(i/num_of_magnets)
                plt.scatter([pend.magnets[i][0]], [pend.magnets[i][1]], s=[150], color=color)
                
            print(f'Animation {0/len(output)*100}% complete', end='')   
            for i in range(len(output)):
                gif_x.append(x[i])
                gif_y.append(y[i])

                l.set_data(gif_x, gif_y)
                
                #plotting balls and sticks for each frame
                balls, = plt.plot(x[i], y[i], 'ro')
                sticks, = plt.plot([0, x[i]], [0, y[i]], '-r')

                # taking frame then clearing plot
                writer.grab_frame()
                balls.remove()
                sticks.remove()
                
                print(f'\rAnimation {i/len(output)*100:.1f}% complete', end='', flush=True)

    elif animation == 'Both':
        l, = plt.plot([], [], '--r')
        gif_x = []
        gif_y = []

        # animating pendulum
        writer = PillowWriter(fps=30, metadata=meta_data)
        with writer.saving(fig, 'TEST_PENDULUM_BOTH_GIF2.gif', 100):
            for i in range(len(output)):
                gif_x.append(x2[i])
                gif_y.append(y2[i])
                l.set_data(gif_x, gif_y)

                #plotting balls and sticks for each frame
                balls, = plt.plot([x1[i], x2[i]], [y1[i], y2[i]], 'ro')
                sticks, = plt.plot([0, x1[i], x2[i]], [0,y1[i], y2[i]], '-r')

                # taking frame then clearing plot
                writer.grab_frame()
                balls.remove()
                sticks.remove()

    if energy:
        for i in output:
            KE = 0.5*pend.m*(x_dot*x_dot + y_dot*y_dot)
            spring_potential = 0.5*pend.g/pend.l*(x*x + y*y)

            e_x, e_y = 0, 0
            for magnet in pend.magnets:
                r = ((x - magnet[0])**2 + (y - magnet[1])**2)**0.5
                distance = (r**2 + pend.min_height**2)**0.5

                e_x += pend.mu*(pend.magnet_strength**2)*(x - magnet[0])/(distance**2)
                e_y += pend.mu*(pend.magnet_strength**2)*(y - magnet[1])/(distance**2)
            magnetic_potential = (e_x**2 + e_y**2)    

            total_energy = KE + spring_potential + magnetic_potential

        plt.show()
        plt.plot(t, total_energy)
        plt.show()


# In[37]:


def run_trials(x_dot=0., y_dot=0., steps_per_period=100, method='Leapfrog', plot_size=4.2,                animation=None, energy=False, trials_per_length=10, magnets=[(0.5, 0)], cmap=plt.get_cmap("jet")):
    """
    Convenience function to run the pendulum in many different positions and plot the results.

    x_dot                  : starting x velocity
    y_dot                  : starting y velocity
    magents                : a list of static magnet positions surrounding the pendulum
    steps_per_period (int) : numer of timesteps per natrual period 
    method (str)           : integration method; one of 'Euler', 'Leapfrog', 'ODEInt'
    animation              : animation style of gif (Real Time, Trace, Both)
    output                 : output of pendulum to animated
    t                      : time array of the pendulum
    energy                 : if true plot the energy
    plot_size              : size of the plot (both axes)
    cmap                   : color scheme of the objects and plot
    trials_per_length      : the number of initial positions considered per unit length
    """
        
    start = timeit.default_timer()            
    pend = MagneticPendulum(x=0, y=0, x_dot=x_dot, y_dot=y_dot, magnets=magnets)

    x_list = np.linspace(-plot_size, plot_size, trials_per_length)
    y_list = np.linspace(-plot_size, plot_size, trials_per_length)
    
    num_of_magnets = len(pend.magnets)
    plt.title('Magnet Final Location From Initial Position')
    plt.figure(figsize=(9,9))
    plt.xlim(-plot_size, plot_size)
    plt.ylim(-plot_size, plot_size)
    color_width = 2*plot_size/trials_per_length
    print(f'\rSimulating position i = {0}, j = {0}.', end='')
    for i in range(len(x_list)):
        x = x_list[i]
        for j in range(len(y_list)):
            y = y_list[j]
            winning_magnet = integrate_and_animate(x=x, y=y, x_dot=x_dot, y_dot=y_dot, steps_per_period=100,                                                         method=method, animation=animation, energy=energy, magnets=magnets,                                                         plot_size=plot_size, cmap=cmap)

            color = cmap(winning_magnet/num_of_magnets)
            rectangle = plt.Rectangle((x_list[i], y_list[j]), color_width, color_width, fc=color)
            plt.gca().add_patch(rectangle)
            print(f'\rSimulating position i = {i}, j = {j}.', end='', flush=True)
    plt.show()
    
    stop = timeit.default_timer()
    print('')
    print(f'Total run-time: {(stop - start):.2f}')
    
run_trials(x_dot=0, y_dot=0., steps_per_period=100, plot_size=4.2,            method='Leapfrog', animation=None, energy=False, trials_per_length=100,            magnets=[(0, -1.5), (-1.299, 0.75), (1.299, 0.75), (0, 3), (3, -3), (-3.5, 0), (3.5, 3.5)],            cmap=plt.get_cmap("jet") )


# In[36]:


integrate_and_animate(x=3.5002, y=0.161, x_dot=0, y_dot=0., steps_per_period=100, plot_size=4.2,            method='Leapfrog', animation='Trace', energy=False,            magnets=[(0, -1.5), (-1.299, 0.75), (1.299, 0.75), (0, 3), (3, -3), (-3.5, 0), (3.5, 3.5)],            cmap=plt.get_cmap("gist_ncar") )


# In[ ]:




