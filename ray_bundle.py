# -*- coding: utf-8 -*-
"""
Class ray_bundle generates a uniform bundle of rays
"""
import numpy as np
import raytracer as rt
import matplotlib.pyplot as plt

class ray_bundle:
    '''
    Initialises a ray bundle with a radius parameter
    Splits the beam into 5 scircles and a ray point in the middle
    '''
    def __init__(self, r):
        rad = np.linspace(0, r, 6)
        x = []
        y = []
        rays = []
        for i in range (0, len(rad)):
            theta = np.linspace(0, 2*np.pi, 6*(i)+1)
            xsin = rad[i] * np.sin(theta)
            ycos = rad[i] * np.cos(theta)
            for j in range (0, 6*(i)+1):
                ray = rt.Ray(pos = [xsin[j], ycos[j], 0], direction = [0, 0, 1])
                rays.append(ray)
            x.append(xsin)
            y.append(ycos)
        self.__rays = rays
    
    def propagate_ray_bundle(self, elements):
        for ray in self.__rays:
            for elem in elements:
                elem.propagate_ray(ray)
    
    def coords(self):
        x = []
        y = []
        z = []
        x_vals = []
        y_vals = []
        z_vals = []
        for ray in self.__rays:
            xvals = ray.x_vals()
            yvals = ray.y_vals()
            zvals = ray.z_vals()
            x.append(xvals)
            y.append(yvals)
            z.append(zvals)
        for i in range (0, len(x[0])):
            x_unpacked = [item[i] for item in x]
            y_unpacked = [item[i] for item in y]
            z_unpacked = [item[i] for item in z]
            x_vals.append(x_unpacked)
            y_vals.append(y_unpacked)
            z_vals.append(z_unpacked)
        return np.array(x_vals), np.array(y_vals), np.array(z_vals)

    def rms_calc(self):
        x, y, z = self.coords()
        x = x[-1]
        y = y[-1]
        sum_magsquare = 0
        for i in range (0, len(x)):
            magsquare = x[i]**2 + y[i]**2
            sum_magsquare += magsquare
        rms = np.sqrt(sum_magsquare/len(x))
        return rms
        
    def zx_plot(self, title=''):
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
            plt.rcParams.update({'font.size': 11})
            x, y, z = self.coords()
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
            x, y, z = self.coords()
            plt.rcParams.update({'font.size': 22})
            fig = plt.figure(figsize = (10, 10), dpi = 600)
            ax = fig.add_subplot(111)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(title)
            plt.grid()
            plt.plot(x[-1], y[-1], 'o')
        
            
    
#%%

#%%
'''
def ray_bundle(r):
    rad = np.linspace(0, r, 6)
    #phasex = phase[0]
    #phasey = phase[1]
    x = []
    y = []
    rays = []
    for i in range (0, len(rad)):
        theta = np.linspace(0, 2*np.pi, 7*(i)-i+1)
        xsin = rad[i] * np.sin(theta)
        ycos = rad[i] * np.cos(theta)
        for j in range (0, 7*(i)-i+1):
            ray = rt.Ray(pos = [xsin[j], ycos[j], 0], direction = [0, 0, 1])
            rays.append(ray)
        x.append(xsin)
        y.append(ycos)
    return rays


def propagate_ray_bundle(ray_bundle, elements):
    for ray in ray_bundle:
        for elem in elements:
            elem.propagate_ray(ray)
            
def ray_bundle_coords(rays):
    x = []
    y = []
    z = []
    x_vals = []
    y_vals = []
    z_vals = []
    for ray in rays:
        xvals = ray.x_vals()
        yvals = ray.y_vals()
        zvals = ray.z_vals()
        x.append(xvals)
        y.append(yvals)
        z.append(zvals)
    for i in range (0, len(x[0])):
        x_unpacked = [item[i] for item in x]
        y_unpacked = [item[i] for item in y]
        z_unpacked = [item[i] for item in z]
        x_vals.append(x_unpacked)
        y_vals.append(y_unpacked)
        z_vals.append(z_unpacked)
        
   
    return np.array(x_vals), np.array(y_vals), np.array(z_vals)
   # return xvals

def rms_calc(x, y):
    sum_magsquare = 0
    for i in range (0, len(x)):
        magsquare = x[i]**2 + y[i]**2
        sum_magsquare += magsquare
    rms = np.sqrt(sum_magsquare/len(x))
    return rms
'''
