import numpy as np
import pandas as pd
from matplotlib import style
import gsw

from pandas.plotting import register_matplotlib_converters
from utility_functions import read_glider_nc

register_matplotlib_converters()

style.use('presentation.mplstyle')
sg579_df = read_glider_nc('data/sg579.nc', glider_id='sg579')
sg620_df = read_glider_nc('data/sg620.nc', glider_id='sg620')
sg637_df = read_glider_nc('data/sg637.nc', glider_id='sg637')
combi = sg579_df.append(sg620_df, sort=False)
glider_df = combi.append(sg637_df, sort=False)

glider_df['dive_limb_ident'] = glider_df['dive']

glider_df.loc[glider_df['direction'] == 1.0, 'dive_limb_ident'] = glider_df['dive_limb_ident'][
                                                                      glider_df['direction'] == 1.0] + 0.5
glider_df.loc[glider_df['glider'] == 'sg620', 'dive_limb_ident'] = glider_df['dive_limb_ident'][
                                                                       glider_df['glider'] == 'sg620'] + 1000
glider_df.loc[glider_df['glider'] == 'sg637', 'dive_limb_ident'] = glider_df['dive_limb_ident'][
                                                                       glider_df['glider'] == 'sg637'] + 2000

ydays = []
for glider_time in glider_df.glider_time:
    t_tuple = glider_time.timetuple()
    ydays.append(
        t_tuple.tm_yday - 1 + t_tuple.tm_hour / 24 + t_tuple.tm_min / (24 * 60) + t_tuple.tm_sec / (24 * 60 * 60))
glider_df['time_yday_utc'] = ydays
glider_df['time_yday_local'] = ydays + np.nanmean(glider_df.lon) / 360
glider_df['hour_of_day'] = 24 * (glider_df.time_yday_local - np.floor(glider_df.time_yday_local))
glider_df['hours_from_noon'] = np.abs(12 - glider_df.hour_of_day)

glider_df['sound_speed'] = gsw.sound_speed(glider_df.abs_salinity, glider_df.temp, glider_df.pressure)

glider_df.loc[glider_df.salinity < 34.5, 'salinity'] = np.nan
glider_df.loc[glider_df.salinity > 37.5, 'salinity'] = np.nan
glider_df.loc[np.isnan(glider_df.abs_salinity), 'salinity'] = np.nan
glider_df.loc[np.isnan(glider_df.salinity), 'abs_salinity'] = np.nan
glider_df.loc[np.isnan(glider_df.salinity), 'conductivity'] = np.nan
glider_df.loc[np.isnan(glider_df.salinity), 'rho'] = np.nan
glider_df.loc[np.isnan(glider_df.salinity), 'sigma0'] = np.nan
glider_df.loc[np.isnan(glider_df.salinity), 'sound_speed'] = np.nan

origin = [-57.338, 14.182]
deg_lat = 110649
glider_df['x'] = (glider_df.lon - origin[0]) * deg_lat * np.cos(np.deg2rad(origin[1]))
glider_df['y'] = (glider_df.lat - origin[1]) * deg_lat
glider_df['along_tsect'] = - np.sqrt(glider_df.x.values ** 2 + glider_df.y.values ** 2)

glider_df['onsite'] = True
glider_df.loc[glider_df['dive_limb_ident'] < 77, 'onsite'] = False
glider_av_df = glider_df.groupby('dive_limb_ident', as_index=False).median()
dive_limb_mean_time = []
for dive in glider_av_df.dive_limb_ident:
    df = glider_df[glider_df.dive_limb_ident == dive]
    dive_limb_mean_time.append((df.glider_time - df.glider_time.min()).mean() + df.glider_time.min())
glider_av_df['glider_time'] = dive_limb_mean_time
glider_av_df['symbol'] = 's'
glider_av_df.loc[glider_av_df['dive_limb_ident'] < 1000, 'symbol'] = 'o'
glider_av_df.loc[glider_av_df['dive_limb_ident'] > 2000, 'symbol'] = '*'
