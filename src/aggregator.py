#!/usr/bin/env python3
"""Parse OSM data
"""

import numpy as np
import numpy
import numpy.linalg
from numpy.linalg import norm as norm
import argparse
import xml.etree.ElementTree as ET
from rtree import index
import matplotlib.pyplot as plt
import logging
from logging import debug, warning
import random
import time

########################################################## DEFS
WAY_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary",
             "unclassified", "residential", "service", "living_street"]
MAX_ARRAY_SZ = 1048576  # 2^20

##########################################################
def create_rtree(points, nodeshash, invsegments, invways):
    """short-description

    Args:
    params

    Returns:
    ret

    Raises:
    """

    t0 = time.time()
    pointsidx = index.Index()

    for pid, p in enumerate(points):
        lat, lon  = p[0:2]
        sid = int(p[2])
        pointsidx.insert(pid, (lat, lon, lat, lon), sid)
        #debug('pid:{}, {}, {}'.format(pid, lat, lon))
    
    debug('R-tree created ({:.3f}s)'.format(time.time() - t0))

    return pointsidx

##########################################################
def test_query(pointsidx, points):
    t0 = time.time()
    for i in range(10):
        idx = np.random.randint(1000)
        x0 = points[idx, 0]
        y0 = points[idx, 0]
        querycoords = (x0, y0, x0, y0)
        list(pointsidx.nearest(querycoords, num_results=1, objects='raw'))
    elapsed = time.time() - t0
    debug(elapsed)

##########################################################
def get_count_by_segment(csvinput, segments, pointstree):
    fh = open(csvinput)
    fh.readline() # Header

    #imageid,n,x,y,t
    nsegments = len(segments.keys())
    counts = np.zeros(nsegments)
    denom = np.zeros(nsegments)
    #denom = np.ones(nsegments)

    from pyproj import Proj, transform
    inProj = Proj(init='epsg:3857')
    outProj = Proj(init='epsg:4326')

    nerrors = 0
    maxcount = 0
    for i, line in enumerate(fh):
        arr = line.split(',')
        count = int(arr[1])
        if not arr[2]:
            nerrors += 1
            continue
        lon = float(arr[2])
        lat = float(arr[3])
        lon, lat = transform(inProj,outProj,lon,lat)
        if count > maxcount: maxcount = count

        querycoords = (lat, lon, lat, lon) 
        sid = list(pointstree.nearest(querycoords, num_results=1, objects='raw'))[0]

        counts[sid] += count
        denom[sid] += 1

        #if i > 10000: break

    for i in range(nsegments):
        if denom[i] > 0:
            counts[i] /= denom[i]
    debug(np.max(counts))
    debug('Max count:{}'.format(maxcount))
    warning('nerrors:{}'.format(nerrors))
    fh.close()
    return counts

##########################################################
def render_matplotlib(nodeshash, ways, crossings, artpoints, queries, avgcounts=[]):
    # Render nodes
    nodes = get_nodes_coords_from_hash(nodeshash)
    #plt.scatter(nodes[:, 1], nodes[:, 0], c='blue', alpha=1, s=20)

    # Render artificial nodes
    #plt.scatter(artpoints[:, 1], artpoints[:, 0], c='blue', alpha=1, s=20)

    # Render ways
    colors = {}
    i = 0
    for wid, wnodes in ways.items():
        i += 1
        r = lambda: random.randint(0,255)
        if avgcounts == np.array([]):
            waycolor = '#%02X%02X%02X' % (r(),r(),r())
        else:
            #waycolor = '#%02X%02X%02X' % (127, 127, int(50+(avgcounts[wid])*10))
            waycolor = 'darkblue'
            alpha = avgcounts[wid] / 6
            if alpha > 1: alpha = 1
        colors[wid] = waycolor
        lats = []; lons = []
        for nodeid in wnodes:
            a, o = nodeshash[nodeid]
            lats.append(a)
            lons.append(o)
        plt.plot(lons, lats, linewidth=3, color=waycolor, alpha=alpha)

    # Render queries
    #for q in queries:
        #plt.scatter(q[1], q[0], linewidth=2, color=colors[q[2]])

    # Render crossings
    crossingscoords = np.ndarray((len(crossings), 2))
    for j, crossing in enumerate(crossings):
        crossingscoords[j, :] = np.array(nodeshash[crossing])

    #plt.scatter(crossingscoords[:, 1], crossingscoords[:, 0], c='black')
    #plt.axis('equal')

    plt.show()

##########################################################
def render_bokeh(nodeshash, ways, crossings, artpoints):
    nodes = get_nodes_coords_from_hash(nodeshash)

    from bokeh.plotting import figure, show, output_file
    TOOLS="hover,pan,wheel_zoom,reset"
    p = figure(tools=TOOLS)

    # render nodes
    p.scatter(nodes[:, 1], nodes[:, 0], size=10, fill_alpha=0.8,
                        line_color=None)

    # render ways
    for wnodes in ways.values():
        r = lambda: random.randint(0, 255)
        waycolor = '#%02X%02X%02X' % (r(),r(),r())
        lats = []; lons = []
        for nodeid in wnodes:
            a, o = nodeshash[nodeid]
            lats.append(a)
            lons.append(o)
        p.line(lons, lats, line_width=2, line_color=waycolor)

    # render crossings
    crossingscoords = np.ndarray((len(crossings), 2))
    for j, crossing in enumerate(crossings):
        crossingscoords[j, :] = np.array(nodeshash[crossing])
    p.scatter(crossingscoords[:, 1], crossingscoords[:, 0], line_color='black')

    output_file("osm-test.html", title="OSM test")

    show(p)  # open a browser
    return


##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputosm', help='Input osm file')
    parser.add_argument('--frontend', choices=['bokeh', 'matplotlib'],
                        help='Front-end vis')
    parser.add_argument('--verbose', help='verbose', action='store_true')

    args = parser.parse_args()

    #if args.verbose:
    if True:
        loglevel = args.verbose if logging.DEBUG else logging.ERROR

    logging.basicConfig(level=loglevel)


    tree = ET.parse(args.inputosm)
    root = tree.getroot() # Tag osm

    ways, invways = parse_ways(root)
    nodeshash = parse_nodes(root, invways)
    ways, invways = filter_out_orphan_nodes(ways, invways, nodeshash)
    crossings = get_crossings(invways)
    segments, invsegments = get_segments(ways, crossings)
    artpoints = evenly_space_segments(segments, nodeshash)

    artpointstree = create_rtree(artpoints, nodeshash, invsegments, invways)

    #for nod, val in nodeshash.items():
        #print(val[0], val[1])
        #break

    queried = []
    #for i in range(1000):
        #nartpoints, _ = artpoints.shape
        #idx = np.random.randint(nartpoints)
        #x = np.random.rand()/1000
        #y = np.random.rand()/1000
        #x0 = artpoints[idx, 0] + x
        #y0 = artpoints[idx, 1] + y
        #querycoords = (x0, y0, x0, y0)
        #segid = list(artpointstree.nearest(querycoords, num_results=1, objects='raw'))[0]
        #queried.append([x0, y0, segid])

    queried = np.array(queried)

    #csvinput = '/home/frodo/projects/timeseries-vis/data/20180901_peds_westvillage_workdays_crs3857_snapped.csv'
    csvinput = '/home/frodo/projects/timeseries-vis/data/20180901_peds_manhattan_workdays_crs3857_snapped.csv'
    mycount = get_count_by_segment(csvinput, segments, artpointstree)
    render_map(nodeshash, segments, crossings, artpoints, queried, mycount, args.frontend)
    
##########################################################
if __name__ == '__main__':
    main()

