#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:02:06 2021

Utillity functions for pysun_slice

@author: jiajia @ Queen's University Belfast
"""

__author__ = 'Jiajia Liu'
__license__ = 'GPLv3'
__version__ = '1.0'
__maintainor__ = 'Jiajia Liu'
__email__ = 'j.liu@qub.ac.uk'


import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

__all__ = ['get_circle_points',
           'lonlat2ij',
           'ij2lonlat']


def get_circle_points(center, radius, resolution=1, npoints=None):
    '''
    given the coordinate of the center and radius of a circle, get all
    coordinates of that cirlce and return it

    Parameters
    ----------
    center : list
        a list with 2 elements representing the (x, y) coordinate
        of the circle.
    radius : float
        radius of the circle
    resolution: float
        space resolution of the circle, in units of pixels
    npoints: integer
        number of points on the circle, if set, resolution will be ignored

    Returns
    -------
    numpy array in shape of (2, N) containing the (x, y) coordinates
    of all points in the circle

    '''
    if np.size(center) != 2:
        raise Exception('center must be the (x, y) coordinate of a point')

    if radius == 0:
        raise Exception('radius cannot be 0')

    if radius < 0:
        radius = np.abs(radius)
        print('radius cannot be negative, use its absolute value now')

    if resolution == 0:
        print('resolution cannot be 0, use 1 now')
        resolution = 1

    if resolution < 0:
        print('resolution cannot be negative, use its absolute value now')
        resolution = np.abs(resolution)

    resolution = float(resolution)

    # number of points on the circle
    n = int(2 * np.pi / (resolution / radius))

    if npoints is not None and npoints != 0:
        if npoints < 0:
            print('npoints cannot be negative, use its absolute value now')
            npoints = np.abs(npoints)
        n = npoints
        resolution = radius * 2 * np.pi / n

    circle = np.zeros([2, n])

    # calculate coordinates
    for i in np.arange(n):
        theta = i * resolution / radius
        circle[:, i] = [center[0] + radius * np.cos(theta),
                        center[1] + radius * np.sin(theta)]

    # close the circle
    circle[:, n-1] = circle[:, 0]

    return circle


def lonlat2ij(lon, lat, smap, coordinate='Stonyhurst'):
    '''

    Given the longitude and latitude of one or a series of points, and their
    corresponding coordinate system, calculate the corresponding x and y
    index in pixel units in the reference sunpy map smap.

    Parameters
    ----------
    lon : float
        a scalar or 1d array that contains the stonyhurst/carrington longtitude
        of all points.
    lat : float
        a scalar or 1d array that contains the stonyhurst/carrington latitude
        of all points.       .
    smap : sunpy.map.Map
        reference observation to be used
    coordinate: string
        must be Stonyhurst or Carrington

    Returns
    -------
    tuple (i, j)

    '''

    if np.shape(lon) != () and np.array(lon).ndim != 1:
        raise Exception('Input longitude must be a scalar or 1D array!')

    if np.shape(lat) != () and np.array(lat).ndim != 1:
        raise Exception('Input longitude must be a scalar or 1D array!')

    if coordinate.lower() != 'stonyhurst' and coordinate.lower() != 'Carrington':
        raise Exception('Coordinate system must be Stronyhurst or Carrington!')

    if np.size(lon) != np.size(lat):
        raise Exception('lon and lat must have the same number of elements!')

    coords = SkyCoord(lon*u.deg, lat*u.deg,
                      frame='heliographic_'+coordinate.lower(),
                      obstime=smap.meta['t_obs'])

    i, j = smap.world_to_pixel(coords)

    return i.value, j.value


def ij2lonlat(i, j, smap, coordinate='Stonyhurst'):
    '''

    Given the pixel coordinate (i, j) of one or a series of points,
    calculate the corresponding longitude and latitude in degrees in the
    reference sunpy map smap and coordinate

    Parameters
    ----------
    i : float
        a scalar or 1d array that contains the x coordinate in pixels
        of all points.
    j : float
        a scalar or 1d array that contains the y coordinate in pixels
        of all points.       .
    smap : sunpy.map.Map
        reference observation to be used
    coordinate: string
        must be Stonyhurst or Carrington

    Returns
    -------
    tuple (longitude, latitude)

    '''

    if np.shape(i) != () and np.array(i).ndim != 1:
        raise Exception('Input x coordinate must be a scalar or 1D array!')

    if np.shape(j) != () and np.array(j).ndim != 1:
        raise Exception('Input y coordiante must be a scalar or 1D array!')

    if coordinate.lower() != 'stonyhurst' and coordinate.lower() != 'Carrington':
        raise Exception('Coordinate system must be Stronyhurst or Carrington!')

    if np.size(i) != np.size(j):
        raise Exception('lon and lat must have the same number of elements!')


    coords = smap.pixel_to_world(i*u.pix, j*u.pix)

    if coordinate.lower() == 'stonyhurst':
        coords = coords.transform_to(frames.HeliographicStonyhurst())
    else:
        coords = coords.transform_to(frames.HeliographicCarrington())

    return coords.lon.value, coords.lat.value