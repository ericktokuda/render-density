"""Plot frames aggregated by time. Blurred scatter plot colored by the counts"""

from bokeh.io import output_notebook, show
from bokeh.plotting import Figure
from functools import partial

import os
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import datetime
import matplotlib.pyplot as plt
import xarray
import PIL
import PIL.ImageFilter
import numpy as np
import logging
import urllib

from datashader.colors import Hot
from matplotlib.cm import viridis, Reds, Blues
from bokeh.palettes import GnBu9

from datashader.bokeh_ext import HoverLayer
from datashader.utils import export_image

output_notebook()


########################################################## GLOBAL VARS
#CMAP = Hot  #Hot, viridis, Reds, ['white', 'darkred'], ['#ffc6c6', 'red']
CMAP = Hot  #Hot, viridis, Reds, ['white', 'darkred'], ['#ffc6c6', 'red']

valrange = [0, 5.5]

DT = datetime.timedelta(minutes=15)
T0 = datetime.time(6, 30)
T1 = datetime.time(22, 0)
WINDOWSIZE = 30
OUTDIR = '/tmp/'
HALFHD = [896, 504]
HD = [1280, 720]
FULLHD = [1920, 1080]
PLOTSIZE = [900, 900]
DISP = PLOTSIZE
INPUTFILE= 'data/data.csv'
# should point to 20170428-peds_workdays_crs3857_snapped.csv
HALFWSIZE = datetime.timedelta(minutes=int(WINDOWSIZE/2))

BLURRADIUS = 0
DOTSIZE = 1

today_str = datetime.datetime.today().strftime('%Y-%m-%d')


##########################################################
def create_base_plot(df, bounds, plotsize, valrange):
    """Create plot based on the input dataframe

    Args:
    df(pandas.dataframe): source data
    bounds(list): xmin, ymin, xmax, ymax
    plotsize(list(2)): width, height
    valrange(list): valmin, valmax to be considered in the coloring (instead of dynamic coloring).

    Returns:
    (bokeh.plotting.Figure, datashader.transfer_functions.Image, xarray.core.dataarray.DataArray)
    """

    (xmin, ymin, xmax, ymax) = bounds

    canvas = ds.Canvas(plot_width=plotsize[0],
                    plot_height=plotsize[1],
                    x_range=(xmin, xmax),
                    y_range=(ymin, ymax))

    agg = canvas.points(df, 'x', 'y', ds.mean('n'))
    myndarray = agg.values
    np.save('/tmp/agg.npy', myndarray)
    
    if valrange:
        img = tf.shade(agg, cmap=CMAP, how='log', alpha=225, span=valrange)
    else:
        img = tf.shade(agg, cmap=CMAP, how='log', alpha=255)
        
    fig = Figure(x_range=(xmin, xmax),
                 y_range=(ymin, ymax),
                 plot_width=DISP[0],
                 plot_height=DISP[1],
                 tools='') #tools='pan,wheel_zoom,reset'
    
    #fig.background_fill_color = 'black'
    fig.toolbar_location = None
    fig.axis.visible = False
    fig.grid.grid_line_alpha = 0
    fig.min_border_left = 0
    fig.min_border_right = 0
    fig.min_border_top = 0
    fig.min_border_bottom = 0

    img = tf.dynspread(img, max_px=DOTSIZE, threshold=1, shape='circle', how='over')
    fig.image_rgba(image=[img.data], x=[xmin], y=[ymin], dw=[xmax-xmin], dh=[ymax-ymin])
    return fig, img, agg

def time_add(t, dt):
    """Sums two datetime.time

    Args:
    t(datetime.time): first element to be added
    dt(datetime.time): second element to be added

    Returns:
    datetime.time: sum t+dt

    """
    return (datetime.datetime.combine(datetime.datetime.today(), t) + dt).time()

def time_subtract(t, dt):
    """Subtract two datetime.time

    Args:
    t(datetime.time): element to be added
    dt(datetime.time): element to be subtracted

    Returns:
    datetime.time: difference t-dt

    """
    return (datetime.datetime.combine(datetime.datetime.today(), t) - dt).time()

def replace_darkalpha_by_whitealpha(pilimg):
    """Replace (0,0,0,0) by (255,255,255,0). Avoid problem of blurring white areas.

    Args:
    pilimg(PIL.image): input image

    Returns:
    PIL.image: image with white alpha

    """
    data = np.array(pilimg)
    red, green, blue, alpha = data.T
    indices = (red == 0) & (green == 0) & (blue == 0) & (alpha == 0)

    data[indices.T] = (255, 255, 255, 0) # Transpose back needed
    newpilimg = PIL.Image.fromarray(data)
    return newpilimg
    

def create_and_save(df, imgout, bounds, plotsize, valrange, blurradius=0):
    """Create the plot and also save to raster

    Args:
    df(pandas.dataframe): source data
    imgout(str): path to the output image
    plotsize(list(2)): width, height
    valrange(list(2)): minvalue, maxvalue to be considered when coloring, instead of dynamic

    Returns:
    (bokeh.plotting.Figure, datashader.transfer_functions.Image,xarray.core.dataarray.DataArray)

    """
    fig, img, datashader_agg = create_base_plot(df, bounds, plotsize, valrange)
    pilimg = img.to_pil()
    if blurradius:
        #pilimg = replace_darkalpha_by_whitealpha(pilimg)
        pilimg = pilimg.filter(PIL.ImageFilter.GaussianBlur(blurradius))
    pilimg.save(imgout + '.png')
    return fig, img, datashader_agg

def create_hover_layer(df, bounds, plotsize, valrange):
    """Create a plot with hover of individual points

    Args:
    df(pandas.dataframe): source data
    bounds(list): xmin, ymin, xmax, ymax
    plotsize(list(2)): width, height
    valrange(list): valmin, valmax to be considered in the coloring (instead of dynamic coloring).
    """
    fig, img, datashader_add = create_base_plot(df, bounds, plotsize, None)
    hover_layer = HoverLayer(agg=datashader_agg, extent=extent,
                             field_name='Average density of people')

    fig.renderers.append(hover_layer.renderer)
    fig.add_tools(hover_layer.tool)
    show(fig)
    
def create_avg_hover_layer(df, bounds, plotsize, valrange):
    """Create a plot with hover indicating the average in a block

    Args:
    df(pandas.dataframe): source data
    bounds(list): xmin, ymin, xmax, ymax
    plotsize(list(2)): width, height
    valrange(list): valmin, valmax to be considered in the coloring (instead of dynamic coloring).
    """
    fig, img, datashader_agg = create_base_plot(df, bounds, plotsize, None)
#     datashader_agg = ndimage.gaussian_filter(datashader_agg.data, sigma=50)
    hover_layer = HoverLayer(field_name='Average density per block',
                         highlight_fill_color='#FFFFFF',
                         highlight_line_color='#FFFFFF',
                         size=30,
                         extent=bounds,
                         agg=datashader_agg,
                         how='mean')

    fig.renderers.append(hover_layer.renderer)
    fig.add_tools(hover_layer.tool)
    show(fig)
    

##########################################################
def main():
    logging.basicConfig(level=logging.DEBUG)
    df = pd.read_csv(INPUTFILE, usecols=['imageid', 'n', 'x', 'y','t'])
    bounds =  [df.x.min(), df.y.min(), df.x.max(), df.y.max()]
    PAD = (df.x.max() - df.x.min()) / 100
    logging.debug("Plot padding: {}".format(PAD))

    bounds[0] = bounds[0] - PAD
    bounds[1] = bounds[1] - PAD
    bounds[2] = bounds[2] + PAD
    bounds[3] = bounds[3] + PAD
    nmax = df.n.max()
    xcenter = (bounds[0]+bounds[2])/2
    ycenter = (bounds[1]+bounds[3])/2

    from pyproj import Proj, transform
    inProj = Proj(init='epsg:3857')
    outProj = Proj(init='epsg:4326')
    xcenter, ycenter = transform(inProj, outProj, xcenter, ycenter)

    def get_mapquest_map():
        zoom = 16
        sz = [1800,1800]
        maptype = 'dark' #light, dark, map
        apikey= '?????'
        req = 'https://www.mapquestapi.com/staticmap/v5/map?key={}&center={},{}&size={},{}&' \
        'type={}&format=png&margin=0&zoom={}'.format(apikey, ycenter, xcenter, sz[0],sz[1], maptype, zoom)
        print(req)
        img = urllib.request.urlopen(req).read()
        with open('/tmp/regionmap.png', 'wb') as fh:
            fh.write(img)


    logging.debug('Generating map of the whole interval')
    create_and_save(df, '/tmp/all', bounds, PLOTSIZE, None, BLURRADIUS)

    return
    get_mapquest_map()

    t = T0
    tb = T1
    while t < tb:
        _beginn = (time_subtract(t, HALFWSIZE)).strftime('%H:%M')
        _end = (time_add(t, HALFWSIZE)).strftime('%H:%M')
        logging.debug('Generating map of {}'.format(_beginn))
        
        filename = '{}_peds_wv_{}'.format(today_str, t.strftime('%H%M'))
        imgout = os.path.join(OUTDIR, filename)
        
        inds = pd.to_datetime(df.t).dt.strftime('%H:%M:').between(_beginn, _end)
        df_filtered = df[inds]
        fig, img, datashader_add = create_and_save(df_filtered, imgout, bounds, PLOTSIZE,
                                                   valrange, BLURRADIUS)
        t = time_add(t, DT)

if __name__ == "__main__":
    main()

