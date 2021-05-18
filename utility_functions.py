from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import cartopy.crs as ccrs



def letterboxes(ax,loc=[0.05,0.85],nobox=False,box_alpha=0.5,offset=0):
    """
    For printing letters onto subplots. Defaults to the upper left corner
    Can take a replacement single location with the loc kwarg
    loc can be specified in pairs of x,y coordinates [[x1,y1],[x2,y2]] etc
    offset to start from another letter offset=2 to start from c etc.
    """
    if np.size(loc)==2:
        loc = np.tile(loc,(len(ax),1))
    loc = np.array(loc)
    for num in range(len(ax)):
        if nobox==False:
            ax[num].text(loc[num,0],loc[num,1],chr(num+97+offset),transform=ax[num].transAxes,
              bbox=dict(facecolor='white', edgecolor='none',alpha=box_alpha))
        else:
            ax[num].text(loc[num,0],loc[num,1],chr(num+97+offset),transform=ax[num].transAxes)


def edge_to_centre(x_in):
    x_out = np.array(x_in)
    return np.nanmean([x_out[1:], x_out[:-1]], axis=0)


def centre_to_edge(x_in):
    x_out = np.array(np.empty(len(x_in) + 1))
    x_out[:-1] = x_in - (x_in[1] - x_in[0]) / 2
    x_out[-1] = x_in[-1] + (x_in[1] - x_in[0]) / 2
    return x_out


def goodcoords(x):
    # transforms from seaglider coordinates into regular lat/lon
    y = np.fix(x / 100) + (x / 100 - np.fix(x / 100)) / 0.6
    return y


coordinate = np.vectorize(goodcoords)


def glider_time_to_neat(glider_time):
    time_cont_blank = np.empty(len(glider_time), dtype=datetime)
    timestamp = []
    for i in range(len(glider_time)):
        day = datetime.fromordinal(int(glider_time[i]))
        dayfrac = timedelta(days=glider_time[i] % 1) - timedelta(days=366)
        time_cont_blank[i] = day + dayfrac
        timestamp.append(pd.Timestamp(time_cont_blank[i]))
    return time_cont_blank, timestamp


def read_glider_nc(glider_netcdf_file, glider_id='sgxxx'):
    glider_nc = xr.open_dataset(glider_netcdf_file)
    df = glider_nc.to_dataframe()
    if 'unnamed' in df.columns:
        df.rename(
            columns={"unnamed": "roll", "unnamed1": "pitch", "unnamed2": "heading"},
            inplace=True,
        )
    df["glider_time"], df.index = glider_time_to_neat(df.index)
    df.loc[df.direction == 1.0, "dive"] += 0.5
    df['glider'] = [glider_id] * len(df.index)
    if 'PAR' in df.columns:
        df.PAR = df.PAR * 1e4  # units after conversion are micro Einstein per cm^2 per sec. Convert to micro Einstein per m^2 per sec
        df[
            'PAR_irr'] = df.PAR * 6.62607015e-34 * 299792458  # Get irradiance units of micro watts per m^2 per sec by multiplying by Plank’s constant and speed of light.
    return df


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(x, y)
    deg = (180 / np.pi) * phi
    if deg < 0:
        deg = 360 + deg
    return (rho, deg)


cart2pol = np.vectorize(cart2pol)

labels = dict(temp='Conservative temperature ($\mathrm{^{\circ}C}$)',
              sal='Absolute salinity ($\mathrm{g\ kg^{-1}}$)',
              oxy='Dissolved oxygen concentration ($\mathrm{\mu mol\ kg^{-1}}$)',
              chl='Chlorophyll $a$ ($\mathrm{mg\ m^{-3}}$)',
              pre='Pressure (dbar)',
              UI='UI ($\mathrm{m^3\ s^{-1}\ km^{-1}}$)',
              vel='Geostrophic velocity ($\mathrm{cm\ s^{-1}}$)',
              scat='Backscatter 650 nm',
              PAR='PAR ($\mathrm{\mu einst\ m^{-2}}$)',
              CDOM='Colored Dissolved Organic Matter',
              scat_scale='Backscatter 650 nm ($\mathrm{10^{-4}}$)',
              sat='Oxygen supersaturation (%)',
              aou='Apparent oxygen utilisation ($\mathrm{\mu mol\ kg^{-1}}$)',
              pden='Potential density ($\mathrm{kg\ m^{-3}}$)',
              depth='Depth (m)')

deg_lat = 110649


def savefig(figname, extension="png"):
    plt.savefig(
        Path('figures') / str(figname + "." + extension),
        format=extension,
        dpi="figure",
        bbox_inches="tight",
    )


cdict0 = {'red': ((0.0, 0.0, 0.0),
                  (1 / 4, 0.0, 1.0),
                  (3 / 4, 1.0, 0.0),
                  (1.0, 0.0, 0.0)),

          'green': ((0.0, 0.0, 0.0),
                    (1 / 4, 0.0, 1.0),
                    (3 / 4, 1.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue': ((0.0, 0.0, 0.0),
                   (1 / 4, 0.0, 1.0),
                   (3 / 4, 1.0, 0.0),
                   (1.0, 0.0, 0.0)),
          }

bw_cmap = LinearSegmentedColormap('tuner_cmap', cdict0)

cdict1 = {'red': ((0.0, 0.0, 0.0),
                  (3 / 8, 0.0, 1.0),
                  (3 / 4, 1.0, 0.0),
                  (1.0, 0.0, 0.0)),

          'green': ((0.0, 0.0, 0.0),
                    (3 / 8, 0.0, 1.0),
                    (5 / 8, 1.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue': ((0.0, 0.0, 0.0),
                   (1 / 4, 0.0, 1.0),
                   (3 / 8, 1.0, 0.0),
                   (1.0, 0.0, 0.0)),
          }

turner_cmap = LinearSegmentedColormap('tuner_cmap', cdict1)


def dist_to_coords(coord_in, distance_shift, shift_units='m'):
    """
    Apples a shift in m to a coordinate in decimal degrees
    :param coord_in: [lon, lat]
    :param distance_shift: [shift_x_m, shift_y_m]
    :return: [lon_shift, lat_shift]
    """
    deg_lat = 111000
    if shift_units == 'km':
        deg_lat = 111
    deg_lon = deg_lat * np.cos(np.deg2rad(coord_in[1]))
    deg_shift = [distance_shift[0] / deg_lon, distance_shift[1] / deg_lat]
    coord_out = np.array(coord_in) + np.array(deg_shift)
    return list(coord_out)


def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3, coord=ccrs.PlateCarree()):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    # Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(coord)
    # Make tmc horizontally centred on the middle of the map,
    # vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)
    # Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    # Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    # Calculate a scale bar length if none has been given
    # (Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000  # in km
        ndim = int(np.floor(np.log10(length)))  # number of digits in number
        length = round(length, -ndim)  # round to 1sf

        # Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']:
                return int(x)
            else:
                return scale_number(x - 10 ** ndim)

        length = scale_number(length)

    # Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    # Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    # Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')
