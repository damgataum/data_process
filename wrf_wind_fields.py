# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap

periods = ['seco',
           # 'chuva'
           ]
inits = ['cfsv2', 'era']

for init in inits:

    print init

    for period in periods:
        print period

        # Loading WRF data

        wrf_ipath = '/home/dam/Documents/dissertation/data/Jaguaribe/' \
                    'aracati_{}_{}/wrfout_d03_2015-10-01_18:00:00.nc' \
            .format(init, period)

        raw_wrf = Dataset(wrf_ipath)

        # Discarding first 6 hours from WRF simulation (spin-up time)
        # wrf_time = raw_wrf.variables['XTIME'][:]  # Minutes since start
        wrf_raw_time = raw_wrf.variables['Times'][6:]  # Date strings

        # Join time strings
        wrf_time = np.array(["".join(t) for t in wrf_raw_time])

        wrf_lat = raw_wrf.variables['XLAT'][0, :, 0]
        wrf_lon = raw_wrf.variables['XLONG'][0, 0, :]
        wrf_hgt = raw_wrf.variables['HGT'][0, :, :]

        wrf_u10 = raw_wrf.variables['U10'][6:, ...]
        wrf_v10 = raw_wrf.variables['V10'][6:, ...]
        wrf_t2 = raw_wrf.variables['T2'][6:, ...]
        wrf_q2 = raw_wrf.variables['Q2'][6:, ...]
        wrf_psfc = raw_wrf.variables['PSFC'][6:, ...]

        raw_wrf.close()

        # print 'time', wrf_time.shape
        # print 'lat', wrf_lat.shape
        # print 'lon', wrf_lon.shape
        # print 'u10', wrf_u10.shape
        # print 'v10', wrf_v10.shape
        # print 't2', wrf_t2.shape

        # Compute Wind Speed and Direction

        wrf_wspd10 = np.sqrt(np.square(wrf_u10) + np.square(wrf_v10))
        wrf_wdir10 = (180 / np.pi) * np.arctan2((-wrf_u10), (-wrf_v10))

        # plt.plot(wrf_wspd[:, 0, 0])
        # plt.show()
        # exit()

        # Estimate Relative Humidity

        wrf_rh2 = wrf_q2 * 100 / ((379.90516 / wrf_psfc)
                                  * np.exp(
            17.2693882 * (wrf_t2 - 273.15) / (wrf_t2 - 35.86)))

        # Convert Temperature to Celsius

        wrf_t2 -= 273.15

        for t, date in enumerate(wrf_time):

            # make 2-d grid of lons, lats
            lons, lats = np.meshgrid(wrf_lon, wrf_lat)
            # make orthographic basemap.
            m = Basemap(resolution='f', projection='tmerc',
                        lat_0=-5., lon_0=-38.,
                        urcrnrlon=wrf_lon[-1], urcrnrlat=wrf_lat[-1],
                        llcrnrlon=wrf_lon[0], llcrnrlat=wrf_lat[0])
            # create figure, add axes
            fig1 = plt.figure(figsize=(8, 10))
            ax = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
            # set desired contour levels.
            clevs = np.arange(0, 15, 1)
            # compute native x,y coordinates of grid.
            x, y = m(lons, lats)
            # define parallels and meridians to draw.
            parallels = np.arange(-90., 91., 0.5)
            meridians = np.arange(0., 361., 0.5)

            # plot Wind Speed contours.

            CS1 = m.contour(x, y, wrf_hgt, np.arange(100, 501, 200),
                            linewidths=0.5, colors='0.1', animated=True)
            plt.clabel(CS1, fontsize=9, inline=1, fmt='%1.0f')

            CS2 = m.contourf(x, y, wrf_wspd10[t, ...], clevs,
                             cmap=plt.cm.jet, animated=True)

            # plot wind vectors on projection grid.
            # transform vectors to projection grid.
            uproj, vproj, xx, yy = \
                m.transform_vector(wrf_u10[t, ...], wrf_v10[t, ...],
                                   wrf_lon, wrf_lat, 21, 21,
                                   returnxy=True, masked=True)
            # now plot.
            Q = m.quiver(xx, yy, uproj, vproj, scale=200)

            # draw coastlines
            m.drawcoastlines(linewidth=1.0)
            m.drawstates(linewidth=1.0)

            # label parallels on right and top
            # meridians on bottom and left
            # labels = [left,right,top,bottom]
            m.drawparallels(parallels, labels=[True, False, False, False],
                            linewidth=0.01)
            m.drawmeridians(meridians, labels=[False, False, False, True],
                            linewidth=0.01)

            # Draw PCD markers

            # pcds = {'moradanova': [-5.136, -38.356],
            #         'jaguaruana': [-4.787, -37.777],
            #         'jaguaribe': [-5.905, -38.628],
            #         'iguatu': [-6.397, -39.270]
            #         }

            x, y = m([-38.356, -37.777, -38.628, -39.270],
                     [-5.136, -4.787, -5.905, -6.397])

            m.plot(x, y, 'ko', markersize=6)

            # add colorbar
            cb = m.colorbar(CS2, "right", size="5%", pad="2%")
            cb.set_label('m/s')
            wind_profile.py
            # set plot title
            ax.set_title(date)
            plt.show()

            exit()
"""
# wrf_time = raw_wrf.variables['XTIME'][:]  # Minutes since start
wrf_time = raw_wrf.variables['Times'][6:]  # Date strings
wrf_lat = raw_wrf.variables['XLAT'][0, :, 0]

wrf_lon = raw_wrf.variables['XLONG'][0, 0, :]
# wrf_z = raw_wrf.variables['HGT'][:]  # Terrain height

#wrf_u = raw_wrf.variables['U'][:]
#wrf_u_lat = raw_wrf.variables['XLAT_U'][:]
#wrf_u_lon = raw_wrf.variables['XLONG_U'][:]

#wrf_v = raw_wrf.variables['V'][:]
#wrf_v_lat = raw_wrf.variables['XLAT_V'][:]
#wrf_v_lon = raw_wrf.variables['XLONG_V'][:]

wrf_u10 = raw_wrf.variables['U10'][6:, ...]
wrf_v10 = raw_wrf.variables['V10'][6:, ...]

#wrf_t = raw_wrf.variables['T'][:]
wrf_t2 = raw_wrf.variables['T2'][6:, ...]

# Computing height
#ph = raw_wrf.variables['PH'][:]
#phb = raw_wrf.variables['PHB'][:]
#wrf_z = (ph + phb)/9.81

raw_wrf.close()

#print wrf_u_lat[0, 0, 0], wrf_u_lat[-1, -1, -1]
#print wrf_u_lon[0, 0, 0], wrf_u_lon[-1, -1, -2]
#print wrf_v_lat[0, 0, 0], wrf_v_lat[-1, -1, -2]
#print wrf_v_lon[0, 0, 0], wrf_v_lon[-1, -1, -1]

#print 'time', wrf_time
#print 'lat', wrf_lat.shape
#print 'lon', wrf_lon.shape
#print 'z', wrf_z.shape
#print 'u', wrf_u.shape
#print 'u10', wrf_u10.shape
#print 'v', wrf_v.shape
#print 'v10', wrf_v10.shape
#print 't', wrf_t.shape
#print 't2', wrf_t2.shape

# Compute Wind Speed and Direction

#wrf_wspd = np.sqrt(np.square(wrf_u) + np.square(wrf_v))
wrf_wspd10 = np.sqrt(np.square(wrf_u10) + np.square(wrf_v10))
#wrf_wdir = (180/np.pi)*np.arctan2((-wrf_u), (-wrf_v))
wrf_wdir10 = (180/np.pi)*np.arctan2((-wrf_u10), (-wrf_v10))

# plt.plot(wrf_time, )
"""
