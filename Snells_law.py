# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 22:55:22 2021

@author: akifc
"""
import numpy as np
import math


def mag(x): 
    return math.sqrt(sum(i**2 for i in x))
"""
def snell_lawv1(p, n1, n2, k1, r):
    
    Use the equation of a sphere:
        x2 + y2 + z2 = r2 
    where r is:
        sqrt(1/absolute(curvature))
    
    Thus we can take the derivative of the sphere with respect to x, y and z
    and sub in the point at which the ray intercepts the sphere
    thus we can find the normal vector direction
    and finally we can normalise it to be a unit vector
    
   
    #we know the derivative with respect to x, y, and z is 2x, 2y and 2z
    surface_normal = 2*p
    surface_normal_k = (1/mag(surface_normal))*surface_normal
    #from dot product formula
    cos_theta1 = np.dot(surface_normal_k, k1)/(mag(surface_normal_k)*mag(k1))
    theta1 = math.acos(cos_theta1)
    """
    
def snells_lawv3(p, n1, n2, k):
    x=p[0]
    y=p[1]
    z=p[2]
    #convert cartesian into spherical polar to find the surface normal vector
    sin_phi = y/np.sqrt(x**2 + y**2)
    cos_phi = x/np.sqrt(x**2 + y**2)
    sin_theta = y/np.sqrt(z**2 + y**2)
    cos_theta = z/np.sqrt(z**2 + y**2)
    normal_k = np.array([sin_theta*cos_phi, sin_theta*sin_phi, cos_theta])
    normalised_k = (1/mag(normal_k))*normal_k
    # then use the dot product formula to find the angle between the vector and the normal
    cos_theta_1 = np.dot(normalised_k, k)
    #1/(mag(normal_k)*mag(k)) would usually be required but the magnitudes are already normalised
    #so not required
    sin_theta_1 = np.sqrt(1.0 - cos_theta_1**2)
    if sin_theta_1 > n2/n1:
        return None
    #elif x == 0 and y == 0:
    elif x == 0 and y == 0 :
        return k
    else:
        n = n1/n2
        sin_theta_2 = n*sin_theta_1
        cos_theta_2 = np.sqrt(1.0 - sin_theta_2**2)
        #from 'https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf'
        # t = incident*n1/n2 + (cos_theta_incidence*n1/n2 - cos_theta_refracted)
        k2 = n*k + (n * cos_theta_1 - cos_theta_2) * normal_k
        return k2
        

def snells_lawv2(p, n1, n2, k, z0):
    x=p[0]
    y=p[1]
    z=p[2]
    #convert cartesian into spherical polar to find the surface normal vector
    r = p-z0
    normalised_k = (1/mag(r))*r
    # then use the dot product formula to find the angle between the vector and the normal
    cos_theta_1 = np.absolute(np.dot(normalised_k, k))
    #1/(mag(normal_k)*mag(k)) would usually be required but the magnitudes are already normalised
    #so not required
    sin_theta_1 = np.sqrt(1.0 - cos_theta_1**2)
    if sin_theta_1 > n2/n1:
        return None
    #elif x == 0 and y == 0:
    elif x == 0 and y == 0 :
        return k
    else:
        n = n1/n2
        sin_theta_2 = n*sin_theta_1
        cos_theta_2 = np.sqrt(1.0 - sin_theta_2**2)
        #from 'https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf'
        # t = incident*n1/n2 + (cos_theta_incidence*n1/n2 - cos_theta_refracted)
        k2 = n*k + (n * cos_theta_1 - cos_theta_2) * normalised_k
        return k2
#%%
x = np.array([[1, 1, 2], [2,2]])

    
    
   
    
    