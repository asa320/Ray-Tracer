# -*- coding: utf-8 -*-
"""
Generates an object class spherical sphere, 
and inherits Optical Element as a base class

The intercept method returns the intercept of a given ray to the surface

Snell's law method returns a new direction vector for a ray which intercepts its surface
depends on the initalised n1 and n2 values it is given

The plot_surface is useful to plot when tracing ray trajectories
"""
import numpy as np
import OpticalElement1 as oe
import math
import matplotlib.pyplot as plt


def mag(x): 
    return math.sqrt(sum(i**2 for i in x))


class SphericalRefraction(oe.OpticalElement):
    '''
    Initializes curvature with 5 parameters:
        z0 is the intercept of the lens with the optical axis
        curv is the curvature of the sphere
        n1 and n2 are the refractive indexes of the sphere outside and inside the medium respectively
        aperture_rad is the raidus for the aperture; it restricts x-y coordinates of the sphere
    '''
    def __init__(self, z0=0, curv = 0.2, n1 = 1, n2 = 3, aperture_rad = 5):
        self.__z0 = np.array([0, 0, z0])
        self.__curv = curv
        self.__n1 = n1
        self.__n2 = n2
        self.__aperture_rad = aperture_rad
        
    def intercept(self, ray):
        p = ray.p()
        k = ray.k()
        if self.__curv == 0.0:
            R = 0
        else:
            R = 1/np.absolute(self.__curv)
        if self.__curv > 0.0:
            O = np.array([self.__z0[0], self.__z0[1], self.__z0[2] + R])
        else:
            O = np.array([self.__z0[0], self.__z0[1], self.__z0[2] - R])
        r = p - O
        if self.__curv > 0.0:
            l = np.dot(-r, np.absolute(k)) - np.sqrt((np.dot(r, np.absolute(k))*np.dot(r, np.absolute(k))) - ((mag(r)*mag(r)) - (R*R)))
        elif self.__curv < 0.0: 
            l = np.dot(-r, np.absolute(k)) + np.sqrt((np.dot(r, np.absolute(k))*np.dot(r, np.absolute(k))) - ((mag(r)*mag(r)) - (R*R)))
        else:
            l = (self.__z0[2] - p[2])/k[2]
        Q = p + (l*k)
        if np.sqrt(Q[0]**2 + Q[1]**2) > self.__aperture_rad:
            return None
        elif self.__curv != 0 and np.sqrt(p[0]**2 + p[1]**2) > R:
            return None
        else:
            return Q
    
    def snells_law(self, ray):
        n1 = self.__n1
        n2 = self.__n2
        p = self.intercept(ray)
        if not isinstance(p, np.ndarray):
            raise ValueError ('No valid intercept for ray')
        k = ray.k()
        x = p[0]
        y = p[1]
        z0 = self.__z0
        if self.__curv == 0.0:
            normalised_k = np.array([0.0, 0.0, -1.0])
        elif self.__curv > 0:
            R = (1/np.absolute(self.__curv))
            O = np.array([z0[0], z0[1], z0[2] + R])
            r = p-O
            normalised_k = (1/mag(r))*r
        else: 
            R = (1/np.absolute(self.__curv))
            O = np.array([z0[0], z0[1], z0[2] - R])
            r = -p+O
            normalised_k = (1/mag(r))*r
        # then use the dot product formula to find the angle between the vector and the normal
        cos_theta_1 = np.absolute(np.dot(normalised_k, k))
        #1/(mag(normal_k)*mag(k)) would usually be required but the magnitudes are already normalised
        #so not required
        sin_theta_1 = np.sqrt(1.0 - cos_theta_1**2)
        if sin_theta_1 > n2/n1:
            return None
        elif x == 0 and y == 0 :
            return k
        else:
            n = n1/n2
            sin_theta_2 = n*sin_theta_1
            cos_theta_2 = np.sqrt(1.0 - sin_theta_2**2)
        #from 'https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf'
        # Vector form of snells law:
        # t = incident*n1/n2 + (cos_theta_incidence*n1/n2 - cos_theta_refracted)
            k2 = n*k + ((n * cos_theta_1) - cos_theta_2) * normalised_k
            normalised_k2 = k2/mag(k2)
            return normalised_k2
        
    def propagate_ray(self, ray):
        p = self.intercept(ray)
        new_k = self.snells_law(ray)
        ray.append(p, new_k)
        return ray.p(), ray.k()
    
    def radius(self):
        if self.__curv == 0.0:
            return 0
        else:
            return 1/np.absolute(self.__curv)
    
    def origin(self):
        if self.__curv > 0:
            return self.__z0[2]+self.radius()
        else:
            return self.__z0[2]-self.radius()
    
    def plot_surface(self):
        plt.rcParams.update({'font.size': 11})
        r = self.radius()
        ar = self.__aperture_rad
        O = self.origin()
        if self.__curv > 0:
            if ar > self.radius():
                theta = np.linspace(np.pi/2, 3*np.pi/2, 1000)
            else:
                theta = np.linspace(np.pi-math.asin(ar/r), np.pi+math.asin(ar/r), 1000)
            z = (r*np.cos(theta)) + O 
            y = (r*np.sin(theta))
        elif self.__curv < 0:
            if ar > r:
                theta = np.linspace(-np.pi/2, np.pi/2, 1000)
            else:
                theta = np.linspace(-math.asin(ar/r), math.asin(ar/r), 1000)
            z = r*np.cos(theta) + O
            y = r*np.sin(theta) 
        else:
            y = np.linspace(-ar, ar, 1000)
            z = np.linspace(O, O, 1000)
        return plt.plot(z, y)

        
#%%
#%%

