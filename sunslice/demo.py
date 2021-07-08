#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 14:01:18 2021

Example codes of using pysun_slice

Make sure you have at least one fits file which could be read by sunpy
in the folder 'data/'

@author: jiajia @ Queen's University Belfast
"""

__author__ = 'Jiajia Liu'
__license__ = 'GPLv3'
__version__ = '1.0'
__maintainor__ = 'Jiajia Liu'
__email__ = 'j.liu@qub.ac.uk'


import glob
from datetime import datetime
import sunpy.map
from astropy import units as u
import astropy.time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from sunslice.sunslice import create_slice, get_time_distance


if __name__ == '__main__':
    # get fits files
    files = sorted(glob.glob('./data/*.fits'))
    # convert to sunpy maps
    maps = sunpy.map.Map(files, sequence=True)

    # reference map which will be used to generate the slice
    refmap = maps[28]

    #define parameters
    stype = 'line'  # type of slice
    width = 20  # width of slice in units of pixels
    resolution = 1 # resolution of slice, in units of pixels
    slice_file = 'slice.hf5'  # name of the file storing information of the slice
    time_distance_file = 'time_distance.hf5' # name of the file storing the time-distance data

    # create the slice
    lslice = create_slice(refmap, width=width, stype=stype, resolution=resolution,
                          outfile=slice_file, clip_interval=[10, 99]*u.percent)

    # generate time distance data
    time_distance = get_time_distance(slice_file, maps, outfile=time_distance_file)

    # show time distance plot
    times = []
    for t in time_distance['time']:
        dummy = astropy.time.Time(t, scale='utc', format='unix')
        times.append(datetime.strptime(dummy.isot, '%Y-%m-%dT%H:%M:%S.%f'))

    distance = time_distance['distance'] / 1000. # Mm
    # average value across the width of the slice
    data = np.nanmean(time_distance['data'], axis=2)
    # running difference
    data = data[1:, :] - data[0:-1, :]

    fig, ax = plt.subplots()
    ax.contourf(times[1:], distance, data.T, cmap='gray', levels=60)
    ax.set_ylabel('Distance (Mm)')
    ax.set_xlabel('Time (UTC)')
    ax.set_title(time_distance['observation'])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.show()
