# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 15:28:10 2021

@author: akifc
"""
import numpy as np
import matplotlib.pyplot as plt
import raytracer as rt
import SphericalRefraction as sr
import output_plane as op
import ray_bundle as rb
import scipy as sp
from scipy.optimize import fmin

#%%
'''
Test Ray class (task 2)
'''
raytest = rt.Ray(pos = [-1, -1, -1], direction = [2, 3, 6])
print(raytest.p())
print(raytest.k())

raytest.append(np.array([0, 0, 1]), np.array([-2, 3, 6]))
print(raytest.vertices())
print(raytest.p())
print(raytest.k())

#%%
'''
Testing spherical surface intercept (task 4)
'''
surface_intercept_test = sr.SphericalRefraction()
# Initialising the surface with the paramters in the class (positive curvature)
raytest_intercept = rt.Ray(pos = [-1, -1, -1], direction = [0, 0, 1])
intercept_test = surface_intercept_test.intercept(raytest_intercept)
print(intercept_test)
# the computed intercept agrees with the value I calculated myself

#%%
# check if the aperture radius works with a new ray with only a z component
# aperture rad initalised with value of 5
# 
ray_zdir = rt.Ray(pos = [3.6, 3.6, 0], direction = [0, 0, 1])
intercept_aperture_test = surface_intercept_test.intercept(ray_zdir)
print(intercept_aperture_test)

#%%
#test negative curvature
raytest_intercept = rt.Ray(pos = [-1, -1, -1], direction = [0, 0, 1])
surface_neg_curv = sr.SphericalRefraction(z0 = 10, curv = -0.02, n1 = 1, n2 = 3, aperture_rad = 5)
neg_intercept = surface_neg_curv.intercept(raytest_intercept)
print(neg_intercept)

#%%
#test 0 curvature:
surface_0 = sr.SphericalRefraction(z0 = 10, curv = 0.0, n1 = 1, n2 = 3, aperture_rad = 5)
intercept0 = surface_0.intercept(raytest_intercept)
print(intercept0)

#%%
'''
Snell's law test (task 7)
'''
snells_law_surface = sr.SphericalRefraction(z0 = 50, curv = 0.01, n1 = 1, n2 =3, aperture_rad = 10)
ray_snell = rt.Ray(pos = [3.6, 3.6, 0], direction = [0, -0.2, 1])
print(ray_snell.k())
snells_law_surface.propagate_ray(ray_snell)
print(ray_snell.k())

#The direction vector changed the way I calculated it would!

#%%
'''
Task 9
Test using mm scale
z0 = 100mm for the spherical refracting object
curv = 0.03
n1 = 1.0
n2 = 1.5
z_output = 250
'''
ray_t9 = rt.Ray(pos = [0.1, 10, 0], direction = [-0.5, -0.5, 30])
ray_t9_2 = rt.Ray(pos = [5, 4, 0], direction = [0.5, 0.5, 30])
ray_t9_3 = rt.Ray(pos = [-1, -3, 0], direction = [1, 1, 20])
surface1 = sr.SphericalRefraction(z0 = 100, curv = 0.03, n1 = 1, n2 = 1.5, aperture_rad = 10)
output1 = op.OutputPlane(250)

elements1 = [surface1, output1]
for elem in elements1:
    elem.propagate_ray(ray_t9)
    elem.propagate_ray(ray_t9_2)
    elem.propagate_ray(ray_t9_3)
ray_t9.zx_plot()
# plot shows the ray propagate to the sphere, then refract and propagate to the output plane
ray_t9_2.zx_plot()
ray_t9_3.zx_plot()
plt.legend(labels = ['Ray 1', 'Ray 2', 'Ray 3', 'Spherical Surface'])
surface1.plot_surface()
plt.title('Task 9 Refraction Test')
plt.xlabel('z position (mm)')
plt.ylabel('x position (mm)')
plt.show()
#plt.savefig('Task 9.png', dpi = 600)
#%%
'''
Task 10 - trace rays parallel to the optical axis
'''
#test multiple rays
ray1 = rt.Ray(pos = [0, 0, 0], direction = [0, 0, 30])
ray2 = rt.Ray(pos = [1, 0, 0], direction = [0, 0, 30])
ray3 = rt.Ray(pos = [2, 0, 0], direction = [0, 0, 30])
ray4 = rt.Ray(pos = [3, 0, 0], direction = [0, 0, 30])
ray5 = rt.Ray(pos = [4, 0, 0], direction = [0, 0, 30])
ray6 = rt.Ray(pos = [5, 0, 0], direction = [0, 0, 30])
ray7 = rt.Ray(pos = [-1, 0, 0], direction = [0, 0, 30])
ray8 = rt.Ray(pos = [-2, 0, 0], direction = [0, 0, 30])
ray9 = rt.Ray(pos = [-3, 0, 0], direction = [0, 0, 30])
ray10 = rt.Ray(pos = [-4, 0, 0], direction = [0, 0, 30])
ray11 = rt.Ray(pos = [-5, 0, 0], direction = [0, 0, 30])

elements = [surface1, output1]
rays = [ray1, ray2, ray3, ray4, ray5, ray6, ray7, ray8, ray9, ray10, ray11]
surface1.plot_surface()
for ray in rays:
    for elem in elements:
        elem.propagate_ray(ray)
    ray.zx_plot()
plt.xlabel('z position (mm)')
plt.ylabel('x position (mm)')
plt.legend(labels = ['Spherical Surface'])
plt.show()
#plt.savefig('Task 10.png', dpi=600)

#estimate position of paraxial focus
#find paraxial focus
ray_paraxial_finder_t10 = rt.Ray([0.1, 0, 0], [0, 0, 1])
spherical_surfaces = [surface1]
for elem in spherical_surfaces:
    elem.propagate_ray(ray_paraxial_finder_t10)
focus_surface1 = ray_paraxial_finder_t10.paraxial_focus()
print('the computed paraxial focus is %f' % focus_surface1)

#theoretical focus for surface 1 using curvature

#%%
'''
task 11
using a negative curvature surface as a test case
going to append the negative of the refracted k vectors and see what happens
'''
surface1_neg = sr.SphericalRefraction(z0 = 100, curv = -0.03, n1 = 1, n2 = 1.5, aperture_rad = 10)
ray1 = rt.Ray(pos = [0, 0, 0], direction = [0, 0, 30])
ray2 = rt.Ray(pos = [1, 0, 0], direction = [0, 0, 30])
ray3 = rt.Ray(pos = [2, 0, 0], direction = [0, 0, 30])
ray4 = rt.Ray(pos = [3, 0, 0], direction = [0, 0, 30])
ray5 = rt.Ray(pos = [4, 0, 0], direction = [0, 0, 30])
ray6 = rt.Ray(pos = [5, 0, 0], direction = [0, 0, 30])
ray7 = rt.Ray(pos = [-1, 0, 0], direction = [0, 0, 30])
ray8 = rt.Ray(pos = [-2, 0, 0], direction = [0, 0, 30])
ray9 = rt.Ray(pos = [-3, 0, 0], direction = [0, 0, 30])
ray10 = rt.Ray(pos = [-4, 0, 0], direction = [0, 0, 30])
ray11 = rt.Ray(pos = [-5, 0, 0], direction = [0, 0, 30])
rays = [ray1, ray2, ray3, ray4, ray5, ray6, ray7, ray8, ray9, ray10, ray11]
for ray in rays:
    surface1_neg.propagate_ray(ray)
#create an output plane farther back
output_neg=op.OutputPlane(-50)
surface1_neg.plot_surface()
for ray in rays:
    output_neg.propagate_ray(ray)
    ray.zx_plot()
plt.xlabel('z position (mm)')
plt.ylabel('x position (mm)')
plt.show()
#plt.savefig('Task 11.png', dpi=600)

#%%
'''
task 12, checking code in main script before defining methods in classes
'''
output_focus = op.OutputPlane(focus_surface1)
output2 = op.OutputPlane(175)
output3 = op.OutputPlane(150)
output4 = op.OutputPlane(125)
output5 = op.OutputPlane(250)
r = np.linspace(0, 5, 5)
for i in range (0, len(r)):
    theta = np.linspace(0, 2*np.pi, (7*(i)-i+1))
    x = r[i] * np.sin(theta)
    y = r[i] * np.cos(theta)
    for j in range (0, 7*(i)-i+1):
        ray_task_12 = rt.Ray(pos = [x[j], y[j], 0], direction = [0, 0, 1])
        elements = [surface1, output4, output3, output2, output_focus, output5]
        for elem in elements:
            elem.propagate_ray(ray_task_12)
        x_vals = ray_task_12.x_vals()
        y_vals = ray_task_12.y_vals()
        z_vals = ray_task_12.z_vals()
        #print(ray_task_12.vertices())
        #plt.plot(x_vals[0], y_vals[0], 'o', color = 'Blue')
        #plt.plot(x_vals[1], y_vals[1], 'o', color = 'Red')
        #plt.plot(x_vals[2], y_vals[2], 'o', color = 'Green')
        #plt.plot(x_vals[3], y_vals[3], 'o', color = 'Blue')
        #plt.plot(x_vals[4], y_vals[4], 'o', color = 'Orange')
        plt.plot(x_vals[5], y_vals[5], 'o', color = 'Black') #output plane at paraxial focus
        #plt.plot(x_vals[6], y_vals[6], 'o', color = 'Black')

        #commented code in the for loop is me testing the spot diagrams at various output plane
#%%
'''
task 12/13 (plotting spot diagrams)
'''
# now using above code but placed in a new 'ray_bundle' class
elements = [surface1, output_focus]
rays = rb.ray_bundle(5)
rays.propagate_ray_bundle(elements)

#I set up a new method in the ray bundle class which allows me to plot the zx plots
rays.zx_plot()
surface1.plot_surface()
plt.xlabel('z position (mm)')
plt.ylabel('x position (mm)')
plt.show()
#plt.savefig('Task 12.png', dpi=600)


#the xy spot diagram plots the xy values for the final vertices of the ray bundle
rays.spot_xy('z = 199.9998mm')
plt.xlabel('x position (mm)')
plt.ylabel('y position (mm)')
plt.show()
#plt.savefig('Task 13.png', dpi=600)


print('rms value is %f' % rays.rms_calc())

#%%
'''
task 15
planoconvex lens part 1
with convex lens first
'''
pcl1 = sr.SphericalRefraction(z0 = 100, curv = 0.02, n1 = 1, n2 = 1.5168, aperture_rad = 10)
ps = sr.SphericalRefraction(z0 = 105, curv = 0.0, n1 = 1.5168, n2 = 1, aperture_rad = 10)

#find paraxial focus
ray_paraxial_finder = rt.Ray([0.1, 0, 0], [0, 0, 1])
spherical_surfaces = [pcl1, ps]
for elem in spherical_surfaces:
    elem.propagate_ray(ray_paraxial_finder)
focus_t15_1 = ray_paraxial_finder.paraxial_focus()


rays_t15 = rb.ray_bundle(5)
elements_t15 = [pcl1, ps, op.OutputPlane(focus_t15_1)]
rays_t15.propagate_ray_bundle(elements_t15)
pcl1.plot_surface()
ps.plot_surface()
plt.legend(labels = ['spherical surface', 'plane surface'])
rays_t15.zx_plot()
plt.xlabel('z position (mm)')
plt.ylabel('x position (mm)')
#plt.savefig('Task 15_1_zx.png', dpi=600)
plt.show()
rays_t15.spot_xy('z = %f' %focus_t15_1)
plt.xlabel('x position (mm)')
plt.ylabel('y position (mm)')
#plt.savefig('Task 15_1_xy.png', bbox_inches='tight', dpi=800)
plt.show()
rms_t15_1 = rays_t15.rms_calc()
print('rms value is %f' % rms_t15_1)

#%%
'''
task 15
planoconvex lens part 2
with convex lens second
'''
pcl2 = sr.SphericalRefraction(z0 = 105, curv = -0.02, n1 = 1.5168, n2 = 1, aperture_rad = 10)
ps2 = sr.SphericalRefraction(z0 = 100, curv = 0.0, n1 = 1, n2 = 1.5168, aperture_rad = 10)
rays_t15_2 = rb.ray_bundle(5)

ray_paraxial_finder = rt.Ray([0.1, 0, 0], [0, 0, 1])
spherical_surfaces = [ps2, pcl2]
for elem in spherical_surfaces:
    elem.propagate_ray(ray_paraxial_finder)
focus_t15_2 = ray_paraxial_finder.paraxial_focus()


elements_t15_2 = [ps, pcl2, op.OutputPlane(focus_t15_2)]
rays_t15_2.propagate_ray_bundle(elements_t15_2)
pcl2.plot_surface()
ps2.plot_surface()
plt.legend(labels = ['plane surface', 'spherical surface'])
rays_t15_2.zx_plot('z = %f' %focus_t15_2)
plt.xlabel('z position (mm)')
plt.ylabel('x position (mm)')
plt.show()
#plt.savefig('Task 15_2_zy.png', dpi=600)

rays_t15_2.spot_xy()
plt.xlabel('x position (mm)')
plt.ylabel('y position (mm)')
plt.show()
#plt.savefig('Task 15_2_xy.png', bbox_inches='tight',  dpi=600)

rms_t15_2 = rays_t15_2.rms_calc()
print('rms value is %f' % rms_t15_2)
#%%
'''
lens optimization
'''

def rms_optimisation(curvs, f):
    #create two surfaces with the input curvatures and propagate rays through
    surf1 = sr.SphericalRefraction(z0 = 100, curv = curvs[0], n1 = 1, n2 = 1.5168, aperture_rad = 100)
    surf2 = sr.SphericalRefraction(z0 = 105, curv = curvs[1], n1 = 1.5168, n2 = 1, aperture_rad = 100)
    #now propagate
    rays = rb.ray_bundle(10)
    output = op.OutputPlane(f)
    elements = [surf1, surf2, output]
    rays.propagate_ray_bundle(elements)
    rms = rays.rms_calc()
    return rms

Initial_guess1 = [0.01, -0.01]

optimal_tnc1 = sp.optimize.fmin_tnc(
    rms_optimisation,
    Initial_guess1,
    approx_grad=True,
    args = [200],
    bounds = [(0,np.inf),(-np.inf,0)]
    )

#%%
print(optimal_tnc1)
print('curvature 1 is %f mm and curvture 2 is %f mm' % (optimal_tnc1[0][0], optimal_tnc1[0][1]))
rms_opt1 = rms_optimisation([optimal_tnc1[0][0], optimal_tnc1[0][1]], 200)


print('rms value is %f' % rms_opt1)

#%%
def find_curv2(curv, u, v, n):
    '''
    Finds curvature of second surface using the thick lens maker formula
    Paramters
        1st surface curvature
        object distance
        image distance
        refractive index of the lens material
    '''
    f = 1 / ((1/u)+(1/v))
    curv2 = curv - (1/((n-1)*f))
    return curv2
def rms_optimisationv2(curv, u, v, d):
    '''
    put in a curv for the first spherical surface and the desired focal length
    '''
    n = 1.5168
    curv2 = find_curv2(curv, u, v, n)
    f = 1 / ((1/u)+(1/v))
    # Always initialise the ray bundle at z=0, so u-(d/2) will be z0 for surf 1
    # u+(d/2) is z0 fr surf 2
    surf1 = sr.SphericalRefraction(z0 = u-(d/2), curv = curv, n1 = 1, n2 = n, aperture_rad = 100)
    surf2 = sr.SphericalRefraction(z0 = u+(d/2), curv = curv2, n1 = n, n2 = 1, aperture_rad = 100)
    rays = rb.ray_bundle(5)
    output = op.OutputPlane(u+f)
    elements = [surf1, surf2, output]
    rays.propagate_ray_bundle(elements)
    rms = rays.rms_calc()
    return rms

u = 200
v = 200
d = 5.0
curv_guess = 0.01
optimalv2 = sp.optimize.fmin_tnc(
    rms_optimisationv2,
    curv_guess,
    approx_grad=True,
    args = [u, v, d],
    bounds = [(0,np.inf)]
    )
#%%
print(optimalv2)
curv1_opt2 = optimalv2[0]

rms_opt2 = rms_optimisationv2(curv1_opt2, u, v, d)
print('rms value is %f' % rms_opt2)

curv2_opt2 = find_curv2(curv1_opt2,  u, v, 1.5168)
print('curvature 1 is %f mm and curvture 2 is %f mm' % (curv1_opt2, curv2_opt2))
#%%
def rms_optimisationv3(curvs, s_o, s_i, d):
    '''
    Parameters
    curvs - list of two curv elements
    s_o - object distacne
    s_i - image distance
    d - distance of the two lenses in the optical plane
    '''
    n = 1.5168
    f = 1 / ((1/s_o) + (1/s_i))
    h1 = -f*(n-1)*d*curvs[0]/n
    h2 = -f*(n-1)*d*curvs[1]/n
    z0_1 = s_o - h1
    z0_2 = z0_1 + d
    surf1 = sr.SphericalRefraction(z0_1, curvs[0], 1, n, 10)
    surf2 = sr.SphericalRefraction(z0_2, curvs[1], n, 1, 10)
    rays = rb.ray_bundle(5)
    focus = z0_2 - h2 + f
    output = op.OutputPlane(focus)
    elements = [surf1, surf2, output]
    rays.propagate_ray_bundle(elements)
    rms = rays.rms_calc()
    return rms

s_o = 200
s_i = 200
d = 5.0
curv_guess = [0.01, -0.00]
optimalv3 = sp.optimize.fmin_tnc(
    rms_optimisationv3,
    curv_guess,
    approx_grad=True,
    args = [s_o, s_i, d],
    bounds = [(0,np.inf), (-np.inf,0)]
    )
#%%
print(optimalv3)
curv1_opt3 = optimalv3[0][0]
curv2_opt3 = optimalv3[0][1]
print('curvature 1 is %f mm and curvture 2 is %f mm' % (curv1_opt3, curv2_opt3))
rms_opt3 = rms_optimisationv3([curv1_opt3, curv2_opt3], s_o, s_i, d)
print('rms value is %f' % rms_opt3)
#%%
rays_t15 = rb.ray_bundle(5)
sr1 = sr.SphericalRefraction(z0=200, curv=curv1_opt3, n1=1, n2=1.5168)
sr2 = sr.SphericalRefraction(z0=205, curv=curv2_opt3, n1=1.5168, n2=1)
elements_t15 = [sr1, sr2, op.OutputPlane(400)]
rays_t15.propagate_ray_bundle(elements_t15)
sr1.plot_surface()
sr2.plot_surface()
plt.legend(labels = ['spherical surface', 'plane surface'])
rays_t15.zx_plot()
plt.xlabel('z position (mm)')
plt.ylabel('x position (mm)')
#plt.savefig('Task 15_1_zx.png', dpi=600)
plt.show()
rays_t15.spot_xy('z = %f' %400)
plt.xlabel('x position (mm)')
plt.ylabel('y position (mm)')
#plt.savefig('Task 15_1_xy.png', bbox_inches='tight', dpi=800)
plt.show()
rms_t15_1 = rays_t15.rms_calc()
print('rms value is %f' % rms_t15_1)
















