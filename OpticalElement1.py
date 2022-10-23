# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 14:11:09 2021

@author: akifc
"""

class OpticalElement:
    def propagate_ray(self, ray):
        "propagate a ray through the optical element"
        raise NotImplementedError()
        