# -*- coding: utf-8 -*-
"""
Output plane at a given z value
"""
import OpticalElement1 as oe
import numpy as np
import math

def mag(x):
    return math.sqrt(sum(i**2 for i in x))

class OutputPlane(oe.OpticalElement):
    '''
    Initialises output plane at specified z value
    '''
    def __init__(self, z):
        self.__z = z
        
    def z(self):
        return self.__z
    
    def intercept(self, ray):
        p = ray.p()
        k = ray.k()
        ray_x = p[0]
        ray_y = p[1]
        ray_z = p[2]
        z_delta = self.z() - ray_z
        kx = k[0]
        ky = k[1]
        kz = k[2]
        new_x = ((z_delta/kz) * kx) + ray_x
        new_y = ((z_delta/kz) * ky) + ray_y
        new_z = self.z()
        new_p = np.array([new_x, new_y, new_z])
        return new_p
    
    def propagate_ray(self, ray):
        new_p = self.intercept(ray)
        ray.append(new_p, ray.k())
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    