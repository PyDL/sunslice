#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:03:52 2021

This file contains a number of functions which help to create a slice in
an image, and generate data for a time-distance plot

@author: jiajia
"""

__author__ = 'Jiajia Liu'
__license__ = 'GPLv3'
__version__ = '1.0'
__maintainor__ = 'Jiajia Liu'
__email__ = 'j.liu@qub.ac.uk'

import numpy as np
import sunpy
import sunpy.map
import sunpy.time
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from sunpy.coordinates import frames
from sunpy.physics.solar_rotation import mapsequence_solar_derotate
import h5py
#from scipy import interpolate
from sunslice.utils import get_circle_points, lonlat2ij


__all__ = ['create_slice',
           'get_time_distance']


def create_slice(file, points=None, width=1, stype='line', angle=None,
                 radius=None, resolution=1, outfile=None, **kwargs):
    '''
    create a slice from a reference image

    Parameters
    ----------
    file : str
        reference file to be read, any file that could be accepted by
        sunpy.map.Map()
    points : tuple, optional
        a tuple that contains the essential information to create the slice
        if stype == 'line', points must be a tuple of ([x0, y0], [x1, y1])
        which are the pixel coordinates of both ends of the linear slice
        if stype == 'circle', points must be a tuple of ([x0, y0], r)
        which are the pixel coordinates of the center of the circle and its
        radius in units of pixels
        if stype == 'sphericalcircle', points must be a tuple of
        ([lon, lat], r), which are the longitude and latitude of the center
        of the spherical circle, and the radius of it. All must be in units
        of degree.
    width : int, optional
        For linear and circular slices, it is the width of the slice in units
        of pixels. The default is 1.
        If stype == 'sphericalcircle', then width is in units of angle
    stype : str, optional
        type of the slice. The default is 'line' for linear slice. It can be
        'circle' for a circular slice in the 2D plane, and 'sphericalcircle'
        for a spherical circle on the Sun.
        For now I have only implemented the above three which are mostly
        used in practive. Furture options may include custom'...
        I wrote all the code in IDL MANY years ago, but don't want to read it
        and change it to Python. I prefer to rethink and write new codes.
    angle : float or string, optional
        For linear slice only
        customised angle (respective to the x-axis) of the slice. If it's a
        string, only accepts "radial" and "tangential" for local radial and
        tangential directions respectively. If it's a float, it is the angle
        of the slice with respect to the x-axis in units of degree
    radius: float, optional
        For cicular slice. it is the customised radius in pixels. For a
        spherical circular slice, it is the customised radius in degrees.
        It will be ignored if the parameter points is given.
    resolution: integer, optional
        resolution of the slice. For a linear slice, resolution should be
        in units of pixels; for a circular slice, resolution should be
        in units of pixels; for a spherical circular slice, resolution should
        be in units of degrees.
    outfile: string, optional
        if set, result of the slice will be saved in outfile which have to be
        a hdf5 file
    **kwargs: sunpy.map.Map.plot() arguments

    Returns
    -------
    dictionary

    The key "all_points" is a numpy array in shape of (w, 2, N) containing
    all points in pixels (for linear and circular slice) or degrees (for a
    spherical circular slice) in the slice. First dimension is the width of
    the slice, the second dimension corresponds to (x, y) in pixels or degrees
    and the third dimension corresponds to the lengh of the slice

    All other keys are self-explained.

    '''
    def onclick(event):
        '''
        response function for mouse click events

        '''
        nonlocal click_points
        if event.button == MouseButton.RIGHT and len(click_points) <= 1:
            coord = [event.xdata, event.ydata]
            click_points = click_points + (coord,)

    #try to load file
    try:
        maps = sunpy.map.Map(file)
    except Exception:
        raise Exception('Input file is not supported by sunpy!')

    try:
        width = int(width)
    except Exception:
        raise Exception('Width of the slice cannot be converted to a number!')

    if width <= 0:
        width = 1

    if width % 2 == 0:
        width = width + 1
        print('width adjusted to an odd number {:d}'.format(width))

    try:
        resolution = int(resolution)
    except Exception:
        raise Exception('Cannot convert resolution to integer')

    if radius is not None:
        if radius == 0:
            raise Exception('radius cannot be 0')
        if radius < 0:
            radius = np.abs(radius)
            print('radius cannot be negative, use its absolute value now')

    # pixel position of the solar disk center
    xd, yd = maps.world_to_pixel(SkyCoord(0*u.degree, 0*u.degree,
                                          frame=maps.coordinate_frame))
    xd = xd.value
    yd = yd.value

    # a linear slice
    if stype == 'line':
        # points is not specified, show the image and allow users to click
        if points is None:
            print('Points not specified, please right click two points ' +
                  'for the start and end points of the slice, then close the ' +
                  'image window')
            ans = 'No'
            while ans[0] != 'Y' and ans[0] != 'y':
                fig = plt.figure()
                ax = plt.subplot(projection=maps)
                maps.plot(**kwargs)
                click_points = ()
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show(block=True)

                # show the points and line
                fig = plt.figure()
                ax = plt.subplot(projection=maps)
                maps.plot(axes=ax, **kwargs)
                points = (click_points[0], click_points[1])
                xs, ys = zip(*points)
                ax.plot(xs*u.pix, ys*u.pix, '--', color='w')
                plt.show(block=False)
                ans = input('Are you happy with this slice? Y/N \n')
                plt.close()


        if np.size(points) != 4:
            raise Exception('Input parameter points has wrong size!')

        # start & end points of the slice
        x0, y0 = points[0]
        x1, y1 = points[1]
        dx = x1 - x0
        dy = y1 - y0
        length = np.sqrt(dx**2 + dy**2)
        n = int(length)

        if angle is not None:
            # distance from disk center
            dis = np.linalg.norm([x0-xd, y0-yd])
            # local radial direction
            if angle == 'radial':
                dx =  length * (x0 - xd) / dis
                dy = length * (y0 - yd) / dis
                print('Slice has been adjusted to local radial direction!')

            # local tangential direction
            elif angle == 'tangential':
                dx = length * (y0 - yd) / dis
                dy = length * (xd - x0) / dis
                print('Slice has been adjusted to local tangential direction!')

            else:
                angle = np.deg2rad(float(angle))
                dx = length * np.cos(angle)
                dy = length * np.sin(angle)
                print('Slice has been adjusted to have a slope of {}'.format(
                      np.tan(angle)))

        points = ([x0, y0], [x0 + dx, y0 + dy])

        # construct all points which will be used in the slice
        result = np.zeros([width, 2, n])
        for w in np.arange(-(width-1)/2, (width+1)/2, 1):
            # new starting point
            xn = x0 + w * dy / length
            yn = y0 - w * dx / length
            result[int(w+(width-1)/2), 0, :] = xn + np.arange(
                                               0, n, resolution) * (dx / length)
            result[int(w+(width-1)/2), 1, :] = yn + np.arange(
                                               0, n, resolution) * (dy / length)
        # return dictionary
        info = {'type': 'line',
                 'unit': 'pixel',
                 'x0': points[0][0],
                 'y0': points[0][1],
                 'x1': points[1][1],
                 'y1': points[1][1],
                 'width': width,
                 'all_points': result}

    # a cicular slice
    elif stype == 'circle':
        # points is not specified, show the image and allow users to click
        if points is None:
            print('Points not specified, please right click two points ' +
                  'for the center of and a point on the circle then close' +
                  ' the image window')
            ans = 'No'
            while ans[0] != 'Y' and ans[0] != 'y':
                fig = plt.figure()
                ax = plt.subplot(projection=maps)
                maps.plot(**kwargs)
                click_points = ()
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show(block=True)

                # show the points and line
                fig = plt.figure()
                ax = plt.subplot(projection=maps)
                maps.plot(axes=ax, **kwargs)
                r = np.linalg.norm([click_points[1][0] - click_points[0][0],
                                        click_points[1][1] - click_points[0][1]])
                if radius is not None:
                    r = radius
                    print('radius specified')
                points = (click_points[0], r)
                circle = get_circle_points(click_points[0], r, resolution)

                xs = circle[0, :]
                ys = circle[1, :]
                ax.plot(xs*u.pix, ys*u.pix, '--', color='w')
                plt.show(block=False)
                ans = input('Are you happy with this slice? Y/N \n')
                plt.close()

        if np.size(points[0]) != 2 or np.size(points[1]) != 1:
            raise Exception('Input parameter points has wrong size!')

        center = points[0]
        radius = points[1]
        circle = get_circle_points(center, radius, resolution)
        n = np.shape(circle)[1]
        if width >= 2 * (radius-1):
            print('width cannot be greater than 2x(radius-1), ' +
                  'adjusted to 2*(radius-1)')
            width = 2 * (radius - 1)
        # construct all points which will be used in the slice
        result = np.zeros([width, 2, n])
        for w in np.arange(-(width-1)/2, (width+1)/2, 1):
            # new circle
            ncircle = get_circle_points(center, radius+w, npoints=n)
            result[int(w+(width-1)/2), 0, :] = ncircle[0, :]
            result[int(w+(width-1)/2), 1, :] = ncircle[1, :]

        info = {'type': 'circle',
                 'unit': 'pixel',
                 'x0': center[0],
                 'y0': center[1],
                 'radius': radius,
                 'width': width,
                 'all_points': result}

    # Spherical circular slice
    elif stype == 'sphericalcircle':
        # points is not specified, show the image and allow users to click
        if points is None:
            print('Points not specified, please right click two points ' +
                  'for the center of and a point on the spherical circle then ' +
                  ' close the image window')
            ans = 'No'
            while ans[0] != 'Y' and ans[0] != 'y':
                fig = plt.figure()
                ax = plt.subplot(projection=maps)
                maps.plot(**kwargs)
                ax.set_autoscale_on(False)
                click_points = ()
                fig.canvas.mpl_connect('button_press_event', onclick)
                plt.show(block=True)

                # show the points and line
                fig = plt.figure()
                ax = plt.subplot(projection=maps)
                maps.plot(axes=ax, **kwargs)
                # longitude and latitude of the points
                center = maps.pixel_to_world(click_points[0][0]*u.pix,
                                             click_points[0][1]*u.pix)
                center = center.transform_to(frames.HeliographicStonyhurst())
                point = maps.pixel_to_world(click_points[1][0]*u.pix,
                                            click_points[1][1]*u.pix)
                point = point.transform_to(frames.HeliographicStonyhurst())
                # spherical distance i.e. radius of the spherical circle
                r = center.separation(point).value

                if radius is not None:
                    r = radius
                    print('radius specified')

                circle = SphericalCircle((center.lon, center.lat), r*u.deg,
                                    resolution = int(360./resolution)).get_xy()
                xs = circle[:, 0]
                ys = circle[:, 1]
                ax.plot(xs, ys, transform=ax.get_transform('heliographic_stonyhurst'))
                plt.show(block=False)
                ans = input('Are you happy with this slice? Y/N \n')
                plt.close()
                points = ([center.lon.value, center.lat.value], r)

        if np.size(points[0]) != 2 or np.size(points[1]) != 1:
            raise Exception('Input parameter points has wrong size!')

        center = points[0]
        radius = points[1]
        circle = SphericalCircle((center[0]*u.deg, center[1]*u.deg), radius*u.deg,
                                 resolution = int(360./resolution)).get_xy()
        n = np.shape(circle)[0]
        if width >= 2 * (radius-1):
            print('width cannot be greater than 2x(radius-1), ' +
                  'adjusted to 2*(radius-1)')
            width = 2 * (radius - 1)
        # construct all points which will be used in the slice
        result = np.zeros([width, 2, n])
        for w in np.arange(-(width-1)/2, (width+1)/2, 1):
            # new circle
            acircle = SphericalCircle((center[0]*u.deg, center[1]*u.deg),
                                      (radius+w)*u.deg, resolution = n-1).get_xy()
            result[int(w+(width-1)/2), 0, :] = acircle[:, 0]
            result[int(w+(width-1)/2), 1, :] = acircle[:, 1]

        info = {'type': 'sphericalcircle',
                 'unit': 'degree',
                 'x0': center[0],
                 'y0': center[1],
                 'radius': radius,
                 'width': width,
                 'all_points': result}
    else:
        raise Exception('Slice type {} not supported yet'.format(stype))

    # show the slice and its both edges
    fig = plt.figure()
    ax = plt.subplot(projection=maps)
    maps.plot(axes=ax, **kwargs)

    if stype == 'sphericalcircle':
        i, j = lonlat2ij(result[int((width-1)/2), 0, :],
                         result[int((width-1)/2), 1, :], maps)
        ax.plot(i*u.pix, j*u.pix, '--', color='w')
        i, j = lonlat2ij(result[0, 0, :], result[0, 1, :], maps)
        ax.plot(i*u.pix, j*u.pix, '--', color='blue')
        i, j = lonlat2ij(result[-1, 0, :], result[-1, 1, :], maps)
        ax.plot(i*u.pix, j*u.pix, '--', color='red')
    elif stype == 'circle' or stype == 'line':
        ax.plot(result[int((width-1)/2), 0, :]*u.pix,
                result[int((width-1)/2), 1, :]*u.pix, '--', color='w')
        plt.show(block=False)
        ax.plot(result[0, 0, :]*u.pix, result[0, 1, :]*u.pix, c='blue')
        ax.plot(result[-1, 0, :]*u.pix, result[-1, 1, :]*u.pix, c='red')
    plt.show()

    # write to file
    if outfile is not None:
        if not outfile.endswith('.hf5') and not outfile.endswith('.hdf5'):
            outfile += '.hf5'

        with h5py.File(outfile, 'w') as f:
            f['all_points'] = info['all_points']
            for key in info:
                if key != 'all_points':
                    f.attrs[key] = info[key]
    return info


def get_time_distance(info, maps, refmap=None, derotate=False, outfile=None):
    '''
    given the information of a slice obtained from create_slice() and a list
    of sunpy maps, obtain the data for the time distance diagram.

    Parameters
    ----------
    info : dictionary or string
        if it is a dictionary, info must be returned from create_slice()
        if it is a string, info must be a hdf5 file written by create_slice()
    maps : list of sunpy.map.Map
        All the maps must have the same size
        if the slice is on-disk, turn on derotation for correct result=
    refmap: sunpy.map.Map, optional
        reference map where the slice was obtained. If not set, the first
        instance of maps will be used.
    derotate: bool, optional
        if set, derotation will be done on the maps
    outfile: string, optional
        if set, result will be saved in outfile which have to be
        a hdf5 file

    Returns
    -------
    dictionary with keys of:
        'data': numpy array with shape of k x n x w where n equals to the length
                of the map list, n equals to the length of the slice, and
                w equals to the width of the slice
        'time': list of strings with m elements for time of each map in the
                list of maps
        'distance': distance in units of km from the start to the end of
                the slice
        'observation': information of the observations
        'dunit': physical unit for distance
        'comments': extra information for understanding the result
        all keys from info except "all_points" are also included.

    '''
    if type(info) is str:
        with h5py.File(info, 'r') as f:
            info = {}
            info['all_points'] = np.array(f['all_points'])
            for key in f.attrs.keys():
                info[key] = f.attrs[key]

    if type(maps) != sunpy.map.mapsequence.MapSequence:
        maps = sunpy.map.Map(maps, sequence=True)

    # refmap is now included in the map sequence
    if derotate:
        if refmap is None:
            print('refmap not specified, use the 0th one in the map sequence' +
                  ' instead!')
            print('Performing derotation...')
            maps = mapsequence_solar_derotate(maps, layer_index=0)
        else:
            maps = sunpy.map.Map([list(maps), refmap], sequence=True)
            meta = maps.all_meta()
            idx = -1
            for i in range(len(maps)):
                if meta[i]['t_obs'] == refmap.meta['t_obs']:
                    idx = i
                    break
            print('Performing derotation...')
            maps = mapsequence_solar_derotate(maps, layer_index=idx)
            maps = list(maps)
            maps.pop(idx)
            maps = sunpy.map.Map(maps, sequence=True)

    # length of slice
    n = np.shape(info['all_points'])[2]
    # width of slice
    w = info['width']
    # length of maps
    k = len(maps)
    result = {'data': np.zeros([k, n, w]),
              'time': [],
              'distance': [0],
              'comments': 'time can be recovered using astropy.time.Time' +
                          '(t, format="unix", scale="utc"). distance is' +
                          ' in units of km',
              'observation': maps[0].meta['telescop'] + ' ' +
                             maps[0].meta['instrume'] + ' ' +
                             str(maps[0].meta['wavelnth']),
              'dunit': 'km'}
    # adding slice information
    for key in info.keys():
        if key != 'all_points':
            result[key] = info[key]

    for k_index in range(k):
        m = maps[k_index]
        t = sunpy.time.parse_time(m.meta['t_obs'])
        #
        result['time'].append(t.unix) #
        # normalize the exptime to 1 second
        d = m.data / m.meta['exptime'] # normalize to 1 second
        ysize, xsize = np.shape(d)
        #x, y = np.arange(xsize), np.arange(ysize)
        # interpolation function, however, doing this will significantly
        # slow down the whole process
        #f = interpolate.interp2d(x, y, d)
        for w_index in range(w):
            xnew = info['all_points'][w_index, 0, :]
            ynew = info['all_points'][w_index, 1, :]
            if info['type'] == 'sphericalcircle':
                # transform longitude latitude to pixel indices
                xnew, ynew = lonlat2ij(xnew, ynew, m)

            xnew = np.array(np.round(xnew), dtype=int)
            ynew = np.array(np.round(ynew), dtype=int)
            xnew[xnew > xsize-1] = xsize - 1
            xnew[xnew < 0 ] = 0
            ynew[ynew > ysize-1] = ysize-1
            ynew[ynew < 0] = 0
            result['data'][k_index, :, w_index] = d[ynew, xnew]

            # for interpolation
            # dummy = np.array([f(x, y) for x, y in zip(xnew, ynew)],
            #                  dtype=float).flatten()
            # result['data'][k_index, :, w_index] = dummy

            if k_index == 0 and w_index == int((w-1)/2):
                disx = (xnew[1:] - xnew[0:-1])
                disx = disx * (m.meta['cdelt1'] * 696340. / m.meta['rsun_obs'])
                disy = (ynew[1:] - ynew[0:-1])
                disy = disy * (m.meta['cdelt2'] * 696340. / m.meta['rsun_obs'])
                dis = np.cumsum(np.sqrt(disx**2 + disy**2))
                result['distance'] = np.insert(dis, 0, 0)

    if outfile is not None:
        if not outfile.endswith('.hf5') and not outfile.endswith('.hdf5'):
            outfile += '.hf5'

        with h5py.File(outfile, 'w') as f:
            f['data'] = result['data']
            f['time'] = result['time']
            f['distance'] = result['distance']
            for key in result:
                if key != 'time' and key != 'data' and key != 'distance':
                    f.attrs[key] = result[key]
    return result
