#!/usr/bin/env python3
"""Parse OSM data
"""

import numpy as np
import argparse
#import osmium
#import shapely.wkb as wkblib
import xml.etree.ElementTree as ET
from rtree import index
import matplotlib.pyplot as plt
import logging
from logging import debug
import random

# Definitions
WAY_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary",
             "unclassified", "residential", "service", "living_street"]

##########################################################
def get_all_nodes(root):
    """Get all nodes in the xml struct

    Args:
    root(ET): root element 

    Returns:
    rtree.index: rtree of the nodes
    """
    nodesidx = index.Index()
    coords = {}

    for child in root:
        if child.tag != 'node': continue
        att = child.attrib
        lat, lon = float(att['lat']), float(att['lon'])
        #print('nodeid:')
        #print(int(att['id']))
        nodesidx.insert(int(att['id']), (lat, lon, lat, lon))
        coords[int(att['id'])] = (lat, lon)

    return nodesidx, coords

##########################################################
def get_all_ways(root, nodescoords):
    """Get all ways in the xml struct

    Args:
    root(ET): root element 

    Returns:
    rtree.index: rtree of the nodes
    """
    ways = {}
    invways = {} # inverted list of ways

    for way in root:
        if way.tag != 'way': continue
        wayid = int(way.attrib['id'])
        isstreet = False
        nodes = []

        nodes = []
        for child in way:
            if child.tag == 'nd':
                nodes.append(int(child.attrib['ref']))
            elif child.tag == 'tag':
                # Found a street segment

                if child.attrib['k'] == 'highway': # TODO: REMOVE IT
                #if child.attrib['k'] == 'highway' and child.attrib['v'] in WAY_TYPES:
                    isstreet = True

        if isstreet:
            ways[wayid]  = nodes
            for node in nodes: invways[node] = wayid

    return ways, invways

##########################################################
def idx2array_nodes(nodes_rtree):
    bounds = nodes_rtree.bounds
    nodeslist = list(nodes_rtree.intersection(bounds, objects=True))
    npoints = len(nodeslist)

    nodes = np.ndarray((npoints, 2))

    for i, node in enumerate(nodeslist):
        nodes[i, 0] = node.bbox[0]
        nodes[i, 1] = node.bbox[1]

    return nodes

##########################################################
def render_map(nodes, ways, frontend='bokeh'):
    if frontend == 'matplotlib':
        render_matplotlib(nodes, ways)
    else:
        render_bokeh(nodes, ways)

##########################################################
def get_nodes_coords_from_hash(nodeshash):
    nnodes = len(nodeshash.keys())
    nodes = np.ndarray((nnodes, 2))

    for j, coords in enumerate(nodeshash.values()):
        nodes[j, 0] = coords[0]
        nodes[j, 1] = coords[1]
    return nodes

##########################################################
def render_matplotlib(nodeshash, ways):
    # render nodes
    nodes = get_nodes_coords_from_hash(nodeshash)
    plt.scatter(nodes[:, 1], nodes[:, 0])

    for wnodes in ways.values():
        r = lambda: random.randint(0,255)
        waycolor = '#%02X%02X%02X' % (r(),r(),r())
        lats = []; lons = []
        for nodeid in wnodes:
            a, o = nodeshash[nodeid]
            lats.append(a)
            lons.append(o)
        plt.plot(lons, lats, linewidth=2, color=waycolor)
    plt.show()

##########################################################
def render_bokeh(nodeshash, ways):
    nodes = get_nodes_coords_from_hash(nodeshash)

    from bokeh.plotting import figure, show, output_file
    TOOLS="hover,pan,wheel_zoom,reset"
    p = figure(tools=TOOLS)

    # render nodes
    p.scatter(nodes[:, 1], nodes[:, 0], size=10, fill_alpha=0.8,
                        line_color=None)

    # render ways
    for wnodes in ways.values():
        r = lambda: random.randint(0,255)
        waycolor = '#%02X%02X%02X' % (r(),r(),r())
        lats = []; lons = []
        for nodeid in wnodes:
            a, o = nodeshash[nodeid]
            lats.append(a)
            lons.append(o)
        p.line(lons, lats, line_width=2, line_color=waycolor)

    output_file("osm-test.html", title="OSM test")

    show(p)  # open a browser
    return

##########################################################
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputosm', help='Input osm file')
    parser.add_argument('--frontend', choices=['bokeh', 'matplotlib'],
                        help='Front end vis')
    parser.add_argument('--verbose', help='verbose', action='store_true')

    args = parser.parse_args()

    args.verbose = True
    if args.verbose:
        loglevel = args.verbose if logging.DEBUG else logging.ERROR

    logging.basicConfig(level=loglevel)

    tree = ET.parse(args.inputosm)
    root = tree.getroot() # Tag osm

    nodestree, nodescoords = get_all_nodes(root)
    ways, invways = get_all_ways(root, nodescoords)

    #nodes = idx2array_nodes(nodesidx)
    render_map(nodescoords, ways, args.frontend)
    
##########################################################
if __name__ == '__main__':
    main()

