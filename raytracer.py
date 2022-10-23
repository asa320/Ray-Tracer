# -*- coding: utf-8 -*-
"""
A Ray class 
Intialises a ray with a 3 coordinate position and direction vector
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def mag(x): 
    return math.sqrt(sum(i**2 for i in x))

class Ray:
    def __init__(self, pos=[0, 0, 0], direction=[1, 1, 1]):
        '''
        Parameters
        pos : list
            3 coordinate position. The default is [0, 0, 0].
        direction : list
            a 3 coordinate direction vector. The default is [1, 1, 1].

        Returns
        An object with a position and direction attribute
        Normalises the direction vector
        '''
        if not isinstance(pos, list):
            raise TypeError ('position must be a list')
        
        if not isinstance(direction, list):
            raise TypeError ('direction must be a list')
            
        if not isinstance(direction, float or int):
            raise TypeError ('direction must be a float or integer')
            
         if not isinstance(pos, float or int):
            raise TypeError ('position must be a float or integer')
        
        self.__pos = [np.array(pos)]
        self.__dir = [(1/mag(direction))*np.array(direction)]
        
    def p(self):
        return self.__pos[-1]
    
    def k(self):
        return self.__dir[-1]
    
    def x_vals(self):
        x = []
        for i in range(0, len(self.__pos)):
            x.append(self.__pos[i][0])
        xvals = np.array(x)
        return xvals
    
    def y_vals(self):
        y = []
        for i in range(0, len(self.__pos)):
            y.append(self.__pos[i][1])
        yvals = np.array(y)
        return yvals
    
    def z_vals(self):
        z = []
        for i in range(0, len(self.__pos)):
            z.append(self.__pos[i][2])
        zvals = np.array(z)
        return zvals
    
    def append(self, new_p, new_k):
        '''
        Parameters
        new_p : np.ndarray
            new position vector to append
        new_k : np.ndarray
            new direction vector to append
        
        Raises
        Exception
            must be an array.
        '''
        if not isinstance(new_p, np.ndarray):
            raise TypeError ('new position is not an array')
            
        if not isinstance(new_k, np.ndarray):
            raise TypeError ('new direction vector is not an array')
            
        #self.__pos = np.append([self.__pos], [new_p], axis=0)
        #self.__dir = np.append([self.__dir], [new_k], axis=0)
        self.__pos.append(new_p)
        self.__dir.append(new_k/mag(new_k))
        
    def vertices(self):
        return self.__pos
    
    def directions(self):
        return self.__dir
    
    def paraxial_focus(self):
        '''
        Important to note
        this only returns the correct focus after the rays have intercepted all the refracting surface
        Using after the output plane will return the wrong result
        '''
        p = self.p()
        k = self.k()
        delta_x = 0-p[0]
        unit_vectors = delta_x/k[0]
        new_p = p+(unit_vectors*k)
        return new_p[2]
    
    def zx_plot(self, title=''):
        '''
        Optional Argument title
        Returns
            plot of z-x values
        
        Raises
        TypeError 
            Title must be a string
        '''
        if not isinstance(title, str):
            raise TypeError ('Title must be in string format')
        else:
            plt.rcParams.update({'font.size': 11})
            x = self.x_vals()
            z = self.z_vals()
            
            plt.plot(z, x)
    
    def spot_xy(self, title=''):
        '''
        Optional Argument title
        Returns
            plot of x-y values
        
        Raises
        TypeError 
            Title must be a string
        '''
        if not isinstance(title, str):
            raise TypeError ('Title must be in string format')
        else:
            x = self.x_vals()
            y = self.y_vals()
            plt.rcParams.update({'font.size': 22})
            fig = plt.figure(figsize = (10, 10), dpi = 100)
            ax = fig.add_subplot(111)
            ax.set_aspect('equal', adjustable='box')
            plt.plot(x, y, 'o')
    
#%%

    
