# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from datetime import datetime, timedelta
from windrose import WindroseAxes
from scipy.stats import pearsonr
from scipy.signal import savgol_filter


def import_wrf_nc(wrf_ipath):
    """

    :param wrf_ipath: 
    :return: wrf_time, wrf_wspd, wrf_wdir, wrf_rh2, wrf_t2
    """

    raw_wrf = Dataset(wrf_ipath)

    # Discarding first 6 hours from WRF simulation (spin-up time)
    # wrf_time = raw_wrf.variables['XTIME'][:]  # Minutes since start
    wrf_raw_time = raw_wrf.variables['Times'][6:]  # Date strings

    # Join time strings
    wrf_time = np.array(["".join(t) for t in wrf_raw_time])

    wrf_lat = raw_wrf.variables['XLAT'][0, :, 0]
    wrf_lon = raw_wrf.variables['XLONG'][0, 0, :]
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

    wrf_wspd = np.sqrt(np.square(wrf_u10) + np.square(wrf_v10))
    wrf_wdir = (180 / np.pi) * np.arctan2((-wrf_u10), (-wrf_v10))

    # plt.plot(wrf_wspd[:, 0, 0])
    # plt.show()
    # exit()

    # Estimate Relative Humidity

    wrf_rh2 = wrf_q2 * 100 / ((379.90516 / wrf_psfc)
                              * np.exp(
        17.2693882 * (wrf_t2 - 273.15) / (wrf_t2 - 35.86)))

    # Convert Temperature to Celsius

    wrf_t2 -= 273.15

    return wrf_time, wrf_lat, wrf_lon, wrf_wspd, wrf_wdir, wrf_t2, wrf_rh2


def nearest_points(lat_array, lon_array, point_lat, point_lon):
    """
    This function returns the 4 nearest points indexes to the desired point.

    :param lat_array: Array of latitudes
    :param lon_array: Array of longitudes
    :param point_lat: Latitude of desired point
    :param point_lon: Longitude of desired point
    :return:
    """
    # Finding nearest latitudes

    lat_idx = (np.abs(lat_array - point_lat)).argmin()

    if lat_array[lat_idx] < point_lat:
        lat_idx1 = lat_idx
        lat_idx2 = lat_idx + 1
    elif lat_array[lat_idx] > point_lat:
        lat_idx1 = lat_idx - 1
        lat_idx2 = lat_idx
    else:
        lat_idx1 = lat_idx
        lat_idx2 = lat_idx

    # Finding nearest longitudes

    lon_idx = (np.abs(lon_array - point_lon)).argmin()

    if lon_array[lon_idx] < point_lon:
        lon_idx1 = lon_idx
        lon_idx2 = lon_idx + 1
    elif lon_array[lon_idx] > point_lon:
        lon_idx1 = lon_idx - 1
        lon_idx2 = lon_idx
    else:
        lon_idx1 = lon_idx
        lon_idx2 = lon_idx

    return lat_idx1, lat_idx2, lon_idx1, lon_idx2


def interpol(var_mtx, lats, lons, point_lat, point_lon):
    """
    This function interpolates gridded data to a desired point.

    :param var_mtx: Variable Matrix - (time, latitude, longitude)
    :param lats: Array of latitudes
    :param lons: Array of longitudes
    :param point_lat: Latitude of desired point
    :param point_lon: Longitude of desired point
    :return: var_point
    """
    lat_idx1, lat_idx2, lon_idx1, lon_idx2 = nearest_points(lats, lons,
                                                            point_lat,
                                                            point_lon)
    var_point = np.full((var_mtx.shape[0]), np.nan)

    for t in range(len(var_mtx)):
        var1 = (point_lon - lons[lon_idx1]) * \
               (var_mtx[t, lat_idx1, lon_idx2] -
                var_mtx[t, lat_idx1, lon_idx1]) / \
               (lons[lon_idx2] - lons[lon_idx1]) + \
               var_mtx[t, lat_idx1, lon_idx1]
        var2 = (point_lon - lons[lon_idx1]) * \
               (var_mtx[t, lat_idx2, lon_idx2] -
                var_mtx[t, lat_idx2, lon_idx1]) / \
               (lons[lon_idx2] - lons[lon_idx1]) + \
               var_mtx[t, lat_idx2, lon_idx1]
        var_point[t] = (point_lat - lats[lat_idx1]) * (var2 - var1) / \
                       (lats[lat_idx2] - lats[lat_idx1]) + var1

    return var_point


def import_pcd(t_0, t_f, pcd_ipath, city, period, variable):
    """

    :param t_0:         Initial date: datetime type
    :param t_f:         Final date: datetime type
    :param pcd_ipath:   Path to pcd data
    :param city:        moradanova, jaguaruana, jaguaribe or iguatu
    :param period:      'seco' or 'chuva'
    :param variable:    'v', 'dir', 'temp', 'rh'
    :return: pcd_dates, pcd_data
    """

    pcd_raw_data = np.loadtxt(
        '{0}/{1}_{2}_{3}.csv'.format(pcd_ipath, city, period, variable),
        dtype=str, skiprows=4, delimiter=';')
    pcd_dates = np.full(len(pcd_raw_data), np.nan, dtype='S19')
    pcd_data = np.full(len(pcd_raw_data), np.nan)

    for r in range(len(pcd_raw_data)):
        pcd_dates[len(pcd_raw_data) - 1 - r] = str(pcd_raw_data[r][0])
        pcd_data[len(pcd_raw_data) - 1 - r] = float(pcd_raw_data[r][1])

    pcd_t0 = \
        np.where(pcd_dates == t_0.strftime("%Y-%m-%d %H:%M:%S"))[0][0]
    pcd_tf = \
        np.where(pcd_dates == t_f.strftime("%Y-%m-%d %H:%M:%S"))[0][0]
    pcd_dates = pcd_dates[pcd_t0:(pcd_tf + 1)]
    pcd_data = pcd_data[pcd_t0:(pcd_tf + 1)]

    return pcd_dates, pcd_data


def import_wrf_ts(wrf_ts_ipath):
    """
    
    :param wrf_ts_ipath: 
    :return: wrf_ts_time, wrf_ts_wspd, wrf_ts_wdir, wrf_ts_t2, wrf_ts_rh2
    """

    raw_wrf = np.loadtxt(wrf_ts_ipath, skiprows=1)

    wrf_ts_raw_time = raw_wrf[:, 1]
    wrf_raw_t2 = raw_wrf[:, 5]
    wrf_q = raw_wrf[:, 6]
    wrf_u = raw_wrf[:, 7]
    wrf_v = raw_wrf[:, 8]
    wrf_p = raw_wrf[:, 9]

    # Computing Relative Humidity

    wrf_ts_rh2_all = wrf_q * 100 / ((379.90516 / wrf_p)
                                    * np.exp(17.2693882 *
                                             (wrf_raw_t2 - 273.15) /
                                             (wrf_raw_t2 - 35.86)
                                             )
                                    )

    # Converting Temperature to Celsius

    wrf_raw_t2 -= 273.15

    # Wind speed and direction

    wrf_ts_wspd_all = np.sqrt(np.square(wrf_u) + np.square(wrf_v))
    wrf_ts_wdir_all = (180 / np.pi) * np.arctan2((-wrf_u), (-wrf_v))

    # Computing hourly means
    wrf_ts_time = []
    wrf_ts_wspd = []
    wrf_ts_wdir = []
    wrf_ts_t2 = []
    wrf_ts_rh2 = []

    for t in range(1, (int(np.fix(np.max(wrf_ts_raw_time))) + 1)):
        wrf_ts_time.append(t)
        t0 = np.where(wrf_ts_raw_time > (t - 1))[0][0]
        tf = np.where(wrf_ts_raw_time <= t)[0][-1]
        wrf_ts_wspd.append(np.mean(wrf_ts_wspd_all[t0:tf + 1]))
        wrf_ts_wdir.append(np.mean(wrf_ts_wdir_all[t0:tf + 1]))
        wrf_ts_t2.append(np.mean(wrf_raw_t2[t0:tf + 1]))
        wrf_ts_rh2.append(np.mean(wrf_ts_rh2_all[t0:tf + 1]))

    wrf_ts_time = wrf_ts_time[5:]
    wrf_ts_wspd = wrf_ts_wspd[5:]
    wrf_ts_wdir = wrf_ts_wdir[5:]
    wrf_ts_t2 = wrf_ts_t2[5:]
    wrf_ts_rh2 = wrf_ts_rh2[5:]

    return wrf_ts_time, wrf_ts_wspd, wrf_ts_wdir, wrf_ts_t2, wrf_ts_rh2


def plot_cfs_era(pcd, pcd_time, wrf_c_ts, wrf_e_ts, wrf_ts_time,
                 wrf_c_nc, wrf_e_nc, wrf_nc_time, variable_y, out_path):

    if len(pcd_time) != len(wrf_ts_time) or len(pcd_time) != len(wrf_time):
        print '\n  Missing obs data. \n'
        return

    # plt.plot(range(len(wrf_ts_time)), wrf_c_ts, 'b:', label='WRF.TS')
    # plt.plot(range(len(wrf_ts_time)), wrf_e_ts, 'r:', label='WRF.TS')
    plt.plot(range(len(wrf_nc_time)), wrf_c_nc, 'b', label='WRF.nc')
    plt.plot(range(len(wrf_nc_time)), wrf_e_nc, 'r', label='WRF.nc')
    plt.plot(range(len(pcd_time)), pcd, 'k', label='OBS')

    plt.legend([# 'WRF_C.TS',
                # 'WRF_E.TS',
                'WRF_CFSv2',
                'WRF_ERA',
                'OBS'])
    plt.title(city.upper())

    x_ticks = range(0, len(pcd_time), 24)
    plt.xticks(x_ticks)
    plt.xlabel('Tempo [horas]')
    plt.ylabel(variable_y)
    # plt.show()
    plt.savefig(out_path)
    plt.close()


def compare_stats(pcd, wrf_c_ts, wrf_e_ts, wrf_c_nc, wrf_e_nc, variable_y):
    # type: (object, object, object, object, object, object) -> object

    # MAE

    # mae_c_ts = np.nansum(np.abs(pcd - wrf_c_ts)) / (len(wrf_c_ts))
    mae_c_nc = np.nansum(np.abs(pcd - wrf_c_nc)) / (len(wrf_c_nc))

    # mae_e_ts = np.nansum(np.abs(pcd - wrf_e_ts)) / (len(wrf_e_ts))
    mae_e_nc = np.nansum(np.abs(pcd - wrf_e_nc)) / (len(wrf_e_nc))

    # MAPE

    pcd[pcd == 0] = 1

    # mape_c_ts = np.nansum(np.abs((pcd - wrf_c_ts) / pcd)) / \
    #             (len(wrf_c_ts)) * 100
    mape_c_nc = np.nansum(np.abs((pcd - wrf_c_nc) / pcd)) / \
                (len(wrf_c_nc)) * 100

    # mape_e_ts = np.nansum(np.abs((pcd - wrf_e_ts) / pcd)) / \
    #             (len(wrf_e_ts)) * 100
    mape_e_nc = np.nansum(np.abs((pcd - wrf_e_nc) / pcd)) / \
                (len(wrf_e_nc)) * 100

    # RMSE

    # rmse_c_ts = np.sqrt(np.nansum((np.power(pcd - wrf_c_ts, 2)) /
    #                               (len(wrf_c_ts))))
    rmse_c_nc = np.sqrt(np.nansum((np.power(pcd - wrf_c_nc, 2)) /
                                  (len(wrf_c_nc))))

    # rmse_e_ts = np.sqrt(np.nansum((np.power(pcd - wrf_e_ts, 2)) /
    #                               (len(wrf_e_ts))))
    rmse_e_nc = np.sqrt(np.nansum((np.power(pcd - wrf_e_nc, 2)) /
                                  (len(wrf_e_nc))))

    # Willmott

    # d_c_ts = 1 - (np.nansum((np.power(pcd - wrf_c_ts, 2))) /
    #               (np.nansum(np.power((np.abs(wrf_c_ts - np.nanmean(pcd)) +
    #                                    np.abs(pcd - np.nanmean(pcd))), 2))))
    d_c_nc = 1 - (np.nansum((np.power(pcd - wrf_c_nc, 2))) /
                  (np.nansum(np.power((np.abs(wrf_c_nc - np.nanmean(pcd)) +
                                       np.abs(pcd - np.nanmean(pcd))), 2))))

    # d_e_ts = 1 - (np.nansum((np.power(pcd - wrf_e_ts, 2))) /
    #               (np.nansum(np.power((np.abs(wrf_e_ts - np.nanmean(pcd)) +
    #                                    np.abs(pcd - np.nanmean(pcd))), 2))))
    d_e_nc = 1 - (np.nansum((np.power(pcd - wrf_e_nc, 2))) /
                  (np.nansum(np.power((np.abs(wrf_e_nc - np.nanmean(pcd)) +
                                       np.abs(pcd - np.nanmean(pcd))), 2))))

    # Pearson

    # print 'T2_C.TS:', pearsonr(pcd_t2, wrf_c_ts_t2)[0]
    # print 'T2_E.TS:', pearsonr(pcd_t2, wrf_e_ts_t2)[0]
    # print 'T2_C.NC:', pearsonr(pcd_t2, wrf_c_nc_t2)[0]
    # print 'T2_E.NC:', pearsonr(pcd_t2, wrf_e_nc_t2)[0]
    # print ' '
    # print 'RH2_C.TS:', pearsonr(pcd_rh2, wrf_c_ts_rh2)[0]
    # print 'RH2_E.TS:', pearsonr(pcd_rh2, wrf_e_ts_rh2)[0]
    # print 'RH2_C.NC:', pearsonr(pcd_rh2, wrf_c_nc_rh2)[0]
    # print 'RH2_E.NC:', pearsonr(pcd_rh2, wrf_e_nc_rh2)[0]
    # print ' '
    # print 'WSPD_C.TS:', pearsonr(pcd_wspd, wrf_c_ts_wspd)[0]
    # print 'WSPD_E.TS:', pearsonr(pcd_wspd, wrf_e_ts_wspd)[0]
    # print 'WSPD_C.NC:', pearsonr(pcd_wspd, wrf_c_nc_wspd)[0]
    # print 'WSPD_E.NC:', pearsonr(pcd_wspd, wrf_e_nc_wspd)[0]

    print variable_y
    print 'INIT      MAE     MAPE    RMSE    Willmott'
    # print 'CFS.ts    {:4.3f}   {:4.3f}   {:4.3f}   {:4.3f}'\
    #     .format(mae_c_ts, mape_c_ts, rmse_c_ts, d_c_ts)
    # print 'ERA.ts    {:4.3f}   {:4.3f}   {:4.3f}   {:4.3f}'\
    #     .format(mae_e_ts, mape_e_ts, rmse_e_ts, d_e_ts)
    print 'CFS      {:4.3f}   {:4.3f}   {:4.3f}   {:4.3f}'\
        .format(mae_c_nc, mape_c_nc, rmse_c_nc, d_c_nc)
    print 'ERA      {:4.3f}   {:4.3f}   {:4.3f}   {:4.3f}'\
        .format(mae_e_nc, mape_e_nc, rmse_e_nc, d_e_nc)
    print ' '

    return


def plt_windrose(wdir, wspd, cor, fig_title, out_path):
    ax = WindroseAxes.from_ax()
    ax.bar(wdir, wspd, normed=True, opening=0.8, color=cor,
           edgecolor='white', bins=1)

    xlabels = ('L', 'NE', 'N', 'NO', 'O', 'SO', 'S', 'SE')

    ax.set_xticklabels(xlabels)
    ax.set_yticks(xrange(0, 51, 10))
    ax.set_yticklabels(xrange(0, 51, 10))

    ax.set_xlabel(fig_title)
    plt.savefig(out_path)
    plt.close()


def scatter_plot(obs_data, sim_data, fig_title, out_path):
    minim = int(np.nanmin(np.c_[list(obs_data), list(sim_data)]))
    maxim = int(np.nanmax(np.c_[list(obs_data), list(sim_data)])) + 1

    plt.plot(obs_data, sim_data, 'ko')
    plt.plot(range(minim, maxim), range(minim, maxim), 'k--')

    plt.xticks(range(minim, maxim))
    plt.yticks(range(minim, maxim))

    plt.title(fig_title)

    plt.xlabel('Observado')
    plt.ylabel('Simulado')

    plt.savefig(out_path)
    plt.close()


def daily_pattern(pcd_data, wrf_c_data, wrf_e_data, time, city, variable_y,
                  out_path):

    wrf_c_patt = np.full((24), np.nan)
    wrf_e_patt = np.full((24), np.nan)
    pcd_patt = np.full((24), np.nan)

    for h, hour in enumerate(xrange(24)):

        hour = ' ' + str(hour).zfill(2) + ':00'

        idxs = []

        for d, date in enumerate(time):

            if hour in date:
                idxs.append(d)

        wrf_c_patt[h] = np.nanmean(wrf_c_data[idxs])
        wrf_e_patt[h] = np.nanmean(wrf_e_data[idxs])
        pcd_patt[h] = np.nanmean(pcd_data[idxs])

    plt.plot(xrange(24), wrf_c_patt, 'b', label='WRF-CFSv2')
    plt.plot(xrange(24), wrf_e_patt, 'r', label='WRF-ERA')
    plt.plot(xrange(24), pcd_patt, 'k', label='OBS')

    plt.legend(['WRF-CFSv2',
                'WRF-ERA',
                'OBS'])

    plt.title(city.upper())

    x_ticks = xrange(24)
    plt.xticks(x_ticks)
    plt.xlabel('Tempo [horas]')
    plt.ylabel(variable_y)
    # plt.show()
    plt.savefig(out_path)
    plt.close()


pcds = {
    'moradanova': [-5.136, -38.356],
    'jaguaruana': [-4.787, -37.777],
    'jaguaribe': [-5.905, -38.628],
    'iguatu': [-6.397, -39.270]
}

periods = {'seco': '10',
           'chuva': '04'
           }
inits = ['cfsv2', 'era']

for period in periods:

    print '\n{}'.format(period.upper())

    # Loading WRF data

    wrf_time, wrf_lat, wrf_lon, wrf_c_wspd, wrf_c_wdir, wrf_c_t2, wrf_c_rh2 = \
        import_wrf_nc('/home/dam/Documents/dissertation/data/Jaguaribe/'
                      'aracati_cfsv2_{}/wrfout_d03_2015-{}-01_18:00:00.nc'
                      .format(period, periods[period]))

    wrf_time, wrf_lat, wrf_lon, wrf_e_wspd, wrf_e_wdir, wrf_e_t2, wrf_e_rh2 = \
        import_wrf_nc('/home/dam/Documents/dissertation/data/Jaguaribe/'
                      'aracati_era_{}/wrfout_d03_2015-{}-01_18:00:00.nc'
                      .format(period, periods[period]))

    # Converting time to local time GMT -3

    for t in xrange(len(wrf_time)):
        dt = datetime.strptime(wrf_time[t], "%Y-%m-%d_%H:%M:%S")
        dt = dt - timedelta(hours=3)
        wrf_time[t] = dt.strftime("%Y-%m-%d %H:%M:%S")

    for city in pcds:

        print '\n{}'.format(city.upper())

        # Interpolating data to PCD location

        # Getting nearest lats and lons and interpolating WRF data

        pcd_lat = pcds[city][0]
        pcd_lon = pcds[city][1]

        # CFSv2
        wrf_c_nc_wspd = interpol(wrf_c_wspd, wrf_lat, wrf_lon, pcd_lat, pcd_lon)
        wrf_c_nc_wdir = interpol(wrf_c_wdir, wrf_lat, wrf_lon, pcd_lat, pcd_lon)
        wrf_c_nc_t2 = interpol(wrf_c_t2, wrf_lat, wrf_lon, pcd_lat, pcd_lon)
        wrf_c_nc_rh2 = interpol(wrf_c_rh2, wrf_lat, wrf_lon, pcd_lat, pcd_lon)

        # ERA-Interin
        wrf_e_nc_wspd = interpol(wrf_e_wspd, wrf_lat, wrf_lon, pcd_lat, pcd_lon)
        wrf_e_nc_wdir = interpol(wrf_e_wdir, wrf_lat, wrf_lon, pcd_lat, pcd_lon)
        wrf_e_nc_t2 = interpol(wrf_e_t2, wrf_lat, wrf_lon, pcd_lat, pcd_lon)
        wrf_e_nc_rh2 = interpol(wrf_e_rh2, wrf_lat, wrf_lon, pcd_lat, pcd_lon)

        # WRF init and final date:

        wrf_t0 = datetime.strptime(wrf_time[0], "%Y-%m-%d %H:%M:%S")
        wrf_tf = datetime.strptime(wrf_time[-1], "%Y-%m-%d %H:%M:%S")

        # Loading Obs data

        pcd_ipath = '/home/dam/Documents/dissertation/data/PCD'

        pcd_wspd_time, pcd_wspd = import_pcd(wrf_t0, wrf_tf,
                                             pcd_ipath, city, period, 'v')
        pcd_wdir_time, pcd_wdir = import_pcd(wrf_t0, wrf_tf,
                                             pcd_ipath, city, period, 'dir')
        pcd_t2_time, pcd_t2 = import_pcd(wrf_t0, wrf_tf,
                                         pcd_ipath, city, period, 'temp')
        pcd_rh2_raw_time, pcd_raw_rh2 = import_pcd(wrf_t0, wrf_tf, pcd_ipath,
                                                   city, period, 'rh')

        # Substituting gaps for NaN

        pcd_rh2_time = []
        pcd_rh2 = np.full(len(wrf_time), np.nan)

        if len(pcd_rh2_raw_time) != len(wrf_time):

            for t, date in enumerate(wrf_time):
                dt = datetime.strptime(date[:-6], "%Y-%m-%d %H")
                pcd_rh2_time.append(dt.strftime("%Y-%m-%d %H:%M:%S"))

                if not dt.strftime("%Y-%m-%d %H:%M:%S") in pcd_rh2_raw_time:
                    pcd_rh2[t] = np.nan
                else:
                    pcd_rh2[t] = pcd_raw_rh2[np.where(
                        pcd_rh2_raw_time == dt.strftime("%Y-%m-%d %H:%M:%S"))]

        else:
            pcd_rh2_time.extend(pcd_rh2_raw_time)
            pcd_rh2[:] = pcd_raw_rh2[:]

        # pcd_dt = datetime.strptime(pcd_wspd_time[0], '%Y-%m-%d %H:%M:%S')
        # wrf_dt = datetime.strptime(wrf_time[0], '%Y-%m-%d_%H:%M:%S')

        # Loading WRF TS data

        # CFSv2
        wrf_ts_time, wrf_c_ts_wspd, wrf_c_ts_wdir, wrf_c_ts_t2, wrf_c_ts_rh2 \
            = import_wrf_ts('/home/dam/Documents/dissertation/data/Jaguaribe/'
                            'aracati_cfsv2_{}/s_{}.d03.TS'.format(period, city))

        # ERA-Interin
        wrf_ts_time, wrf_e_ts_wspd, wrf_e_ts_wdir, wrf_e_ts_t2, wrf_e_ts_rh2 \
            = import_wrf_ts('/home/dam/Documents/dissertation/data/Jaguaribe/'
                            'aracati_era_{}/s_{}.d03.TS'.format(period, city))

        # Plotting figures

        # T 2m
        plot_cfs_era(pcd_t2, pcd_t2_time,
                     wrf_c_ts_t2, wrf_e_ts_t2, wrf_ts_time,
                     wrf_c_nc_t2, wrf_e_nc_t2, wrf_time,
                     u'Temperatura a 2m [°C]',
                     'figures/plot_wrf_pcd_t2_{}_{}.png'.format(city, period))

        # Wind speed

        plot_cfs_era(pcd_wspd, pcd_wspd_time,
                     wrf_c_ts_wspd, wrf_e_ts_wspd, wrf_ts_time,
                     wrf_c_nc_wspd, wrf_e_nc_wspd, wrf_time,
                     'Velocidade do vento a 10m [m/s]',
                     'figures/plot_wrf_pcd_wspd_{}_{}.png'.format(city, period))

        plot_cfs_era(pcd_rh2, pcd_rh2_time,
                     wrf_c_ts_rh2, wrf_e_ts_rh2, wrf_ts_time,
                     wrf_c_nc_rh2, wrf_e_nc_rh2, wrf_time,
                     u'Umidade Relativa a 2m [%]',
                     'figures/plot_wrf_pcd_rh2_{}_{}.png'.format(city, period))

        compare_stats(pcd_t2,
                      wrf_c_ts_t2, wrf_e_ts_t2,
                      wrf_c_nc_t2, wrf_e_nc_t2,
                      u'Temperatura a 2m [°C]')

        compare_stats(pcd_wspd,
                      wrf_c_ts_wspd, wrf_e_ts_wspd,
                      wrf_c_nc_wspd, wrf_e_nc_wspd,
                      'Velocidade do vento a 10m [m/s]')

        compare_stats(pcd_rh2,
                      wrf_c_ts_rh2, wrf_e_ts_rh2,
                      wrf_c_nc_rh2, wrf_e_nc_rh2,
                      u'Umidade Relativa a 2m [%]')

        # Wind Rose plot

        plt_windrose(wrf_c_nc_wdir, wrf_c_nc_wspd, 'b',
                     'WRF - CFSv2\n{} - {}'.format(city.upper(),
                                                   period.upper()),
                     'figures/windrose_wrf_c_{}_{}.png'.format(city, period))

        plt_windrose(wrf_e_nc_wdir, wrf_e_nc_wspd, 'g',
                     'WRF - ERA-Interim\n{} - {}'.format(city.upper(),
                                                         period.upper()),
                     'figures/windrose_wrf_e_{}_{}.png'.format(city, period))

        plt_windrose(pcd_wdir, pcd_wspd, 'r',
                     'PCD\n{} - {}'.format(city.upper(), period.upper()),
                     'figures/windrose_pcd_{}_{}.png'.format(city, period))

        # Scatter plot

        # scatter_plot(pcd_t2, wrf_ts_t2,
        #              'scatter_t2_ts_{}_{}.png'.format(init, period))
        # scatter_plot(pcd_wspd, wrf_ts_wspd,
        #              'scatter_wspd_ts_{}_{}.png'.format(init, period))
        scatter_plot(pcd_t2, wrf_c_nc_t2,
                     'Temperatura a 2m (WRF-CFSv2)\n{} - {}'
                     .format(city.upper(), period.upper()),
                     'figures/scatter_t2_cfs_{}_{}.png'.format(city, period))
        scatter_plot(pcd_rh2, wrf_c_nc_rh2,
                     'Humidade Relativa a 2m (WRF-CFSv2)\n{} - {}'
                     .format(city.upper(), period.upper()),
                     'figures/scatter_rh2_cfs_{}_{}.png'.format(city, period))
        scatter_plot(pcd_wspd, wrf_c_nc_wspd,
                     'Velocidade do vento a 10m (WRF-CFSv2)\n{} - {}'
                     .format(city.upper(), period.upper()),
                     'figures/scatter_wspd_cfs_{}_{}.png'.format(city, period))

        scatter_plot(pcd_t2, wrf_e_nc_t2,
                     'Temperatura a 2m (WRF-ERA-Interim)\n{} - {}'
                     .format(city.upper(), period.upper()),
                     'figures/scatter_t2_era_{}_{}.png'.format(city, period))
        scatter_plot(pcd_rh2, wrf_e_nc_rh2,
                     'Humidade Relativa a 2m (WRF-ERA-Interim)\n{} - {}'
                     .format(city.upper(), period.upper()),
                     'figures/scatter_rh2_era_{}_{}.png'.format(city, period))
        scatter_plot(pcd_wspd, wrf_e_nc_wspd,
                     'Velocidade do vento a 10m (WRF-ERA-Interim)\n{} - {}'
                     .format(city.upper(), period.upper()),
                     'figures/scatter_wspd_era_{}_{}.png'.format(city, period))

        # Daily Pattern

        daily_pattern(pcd_t2, wrf_c_nc_t2, wrf_e_nc_t2, wrf_time,
                      city, u'Temperatura a 2m [°C]',
                      'figures/daily_pattern_t2_{}_{}.png'.format(city, period))

        daily_pattern(pcd_wspd, wrf_c_nc_wspd, wrf_e_nc_wspd, wrf_time,
                      city, 'Velocidade do vento a 10m [m/s]',
                      'figures/daily_pattern_wspd_{}_{}.png'.format(city, period))

        daily_pattern(pcd_rh2, wrf_c_nc_rh2, wrf_e_nc_rh2, wrf_time,
                      city, u'Umidade Relativa a 2m [%]',
                      'figures/daily_pattern_rh2_{}_{}.png'.format(city, period))
