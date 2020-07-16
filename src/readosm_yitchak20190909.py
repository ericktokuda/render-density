#!/usr/bin/env python3
"""Parse OSM data
"""
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.style.use('ggplot')

import os
import numpy as np
import numpy
import numpy.linalg
from numpy.linalg import norm as norm
import argparse
import xml.etree.ElementTree as ET
from rtree import index
import logging
from logging import debug, warning
import random
import time
import pickle

import concurrent.futures
import scipy.spatial
from pyproj import Proj, transform


########################################################## DEFS
WAY_TYPES = ["motorway", "trunk", "primary", "secondary", "tertiary",
             "unclassified", "residential", "service", "living_street"]
MAX_ARRAY_SZ = 1048576  # 2^20

intersection_delta = 1e-4

##########################################################
def parse_nodes(root, invways):
    """Get all nodes in the xml struct

    Args:
    root(ET): root element 
    invways(dict): inverted list of ways, i.e., node as key and list of way ids as values

    Returns:

    """
    t0 = time.time() 
    valid = invways.keys()
    nodeshash = {}

    for child in root:
        if child.tag != 'node': continue # not node
        if int(child.attrib['id']) not in valid: continue # non relevant node

        att = child.attrib
        lat, lon = float(att['lat']), float(att['lon'])
        nodeshash[int(att['id'])] = (lat, lon)
    debug('Found {} (traversable) nodes ({:.3f}s)'.format(len(nodeshash.keys()),
                                                          time.time() - t0))

    return nodeshash

##########################################################
def parse_ways(root):
    """Get all ways in the xml struct

    Args:
    root(ET): root element 

    Returns:
    dict of list: hash of wayids as key and list of nodes as values;
    dict of list: hash of nodeid as key and list of wayids as values;
    """

    t0 = time.time()
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
                if child.attrib['k'] == 'highway' and child.attrib['v'] in WAY_TYPES:
                    isstreet = True

        if isstreet:
            ways[wayid] = nodes

            for node in nodes:
                if node in invways.keys(): invways[node].append(wayid)
                else: invways[node] = [wayid]

    debug('Found {} ways ({:.3f}s)'.format(len(ways.keys()), time.time() - t0))
    return ways, invways

##########################################################
def render_map(nodeshash, segments, crossings, artpoints,crossing_points,counts,intersection_counts, show, outdir):
    if show:
        debug('Rendering map to screen')
        render_bokeh(nodeshash, segments, crossings, artpoints, counts)
    else:
        debug('Rendering map to image')
        render_matplotlib(nodeshash, segments, crossings, artpoints,crossing_points,counts,intersection_counts,outdir)

##########################################################
def get_nodes_coords_from_hash(nodeshash):
    """Get nodes coordinates and discard nodes ids information
    Args:
    nodeshash(dict): nodeid as key and (x, y) as value

    Returns:
    np.array(n, 2): Return a two-column table containing all the coordinates
    """

    nnodes = len(nodeshash.keys())
    nodes = np.ndarray((nnodes, 2))

    for j, coords in enumerate(nodeshash.values()):
        nodes[j, 0] = coords[0]
        nodes[j, 1] = coords[1]
    return nodes

##########################################################
def get_crossings(invways):
    """Get crossings

    Args:
    invways(dict of list): inverted list of the ways. It s a dict of nodeid as key and
    list of wayids as values

    Returns:
    set: set of crossings
    """

    crossings = set()
    for nodeid, waysids in invways.items():
        if len(waysids) > 1:
            crossings.add(nodeid)
    return crossings

##########################################################
def filter_out_orphan_nodes(ways, invways, nodeshash):
    """Check consistency of nodes in invways and nodeshash and fix them in case
    of inconsistency
    It can just be explained by the *non* exitance of nodes, even though they are
    referenced inside ways (<nd ref>)

    Args:
    invways(dict of list): nodeid as key and a list of wayids as values
    nodeshash(dict of 2-uple): nodeid as key and (x, y) as value
    ways(dict of list): wayid as key and an ordered list of nodeids as values

    Returns:
    dict of list, dict of lists
    """

    ninvways = len(invways.keys())
    nnodeshash = len(nodeshash.keys())
    if ninvways == nnodeshash: return ways, invways

    validnodes = set(nodeshash.keys())

    # Filter ways
    for wayid, nodes in ways.items():
        newlist = [ nodeid for nodeid in nodes if nodeid in validnodes ]
        ways[wayid] = newlist
        
    # Filter invways
    invwaysnodes = set(invways.keys())
    invalid = invwaysnodes.difference(validnodes)

    for nodeid in invalid:
        del invways[nodeid]

    ninvways = len(invways.keys())
    debug('Filtered {} orphan nodes.'.format(ninvways - nnodeshash))
    return ways, invways

##########################################################
def get_segments(ways, crossings):
    """Get segments, given the ways and the crossings

    Args:
    ways(dict of list): hash of wayid as key and list of nodes as values
    crossings(set): set of nodeids in crossings

    Returns:
    dict of list: hash of segmentid as key and list of nodes as value
    dict of list: hash of nodeid as key and list of sids as value
    """

    t0 = time.time() 
    segments = {}
    invsegments = {}

    sid = 0 # segmentid
    segment = []
    for w, nodes in ways.items():
        if not crossings.intersection(set(nodes)):
            segments[sid] = nodes
            sid += 1
            continue

        segment = []
        for node in nodes:
            segment.append(node)

            if node in crossings and len(segment) > 1:
                segments[sid] = segment
                for snode in segment:
                    if snode in invsegments: invsegments[snode].append(sid) 
                    else: invsegments[snode] = [sid]
                segment = [node]
                sid += 1

        segments[sid] = segment # Last segment
        for snode in segment:
            if snode in invsegments: invsegments[snode].append(sid) 
            else: invsegments[snode] = [sid]
        sid += 1

    debug('Found {} segments ({:.3f}s)'.format(len(segments.keys()), time.time() - t0))
    return segments, invsegments

##########################################################
def evenly_space_segment(segment, nodeshash, epsilon):
    """Evenly space one segment

    Args:
    segment(list): nodeids composing the segment
    nodeshash(dict of list): hash with nodeid as value and 2-uple as value

    Returns:
    coords(ndarray(., 2)): array of coordinates
    """
    #debug(segment)
    prevnode = np.array(nodeshash[segment[0]])
    points = [prevnode]

    for nodeid in segment[1:]:
        node = np.array(nodeshash[nodeid])
        d = norm(node - prevnode)
        if d < epsilon:
            points.append(node)
            prevnode = node
            continue

        nnewnodes = int(d / epsilon)
        direc = node - prevnode
        direc = direc / norm(direc)

        cur = prevnode
        for i in range(nnewnodes):
            newnode = cur + direc*epsilon
            points.append(newnode)
            cur = newnode
        prevnode = node
    return np.array(points)



##########################################################
def evenly_space_segments(segments, nodeshash, epsilon=0.0001):
    """Evenly space all segments and create artificial points

    Args:
    segments(dict of list): hash of segmentid as key and list of nodes as values
    nodeshash(dict of list): hash with nodeid as value and 2-uple as value

    Returns:
    points(ndarray(., 3)): rows represents points and first and second columns
    represent coordinates and third represents the segmentid
    """

    t0 = time.time()
    points = np.ndarray((MAX_ARRAY_SZ, 3))
    idx = 0
    for sid, segment in segments.items():
        coords = evenly_space_segment(segment, nodeshash, epsilon)
        n, _ = coords.shape
        points[idx:idx+n, 0:2] = coords
        points[idx:idx+n, 2] = sid
        idx = idx + n
    debug('New {} support points ({:.3f}s)'.format(idx, time.time() - t0))
    return points[:idx, :]

###########################################################
def get_crossing_points(crossings,nodeshash):
    locations = []
    
    for nid in crossings:
        locations.append(np.array(nodeshash[nid]))
    
    return np.array(locations)
    

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

def get_count_by_segment(csvinput, segments,artpoints,crossing_points):
    
    
    t0 = time.time()
    artpointstree = scipy.spatial.cKDTree(artpoints[:,:2]) 
    crossing_pointstree = scipy.spatial.cKDTree(crossing_points[:,:2]) 

    debug('cKDTree created ({:.3f}s)'.format(time.time() - t0))
    
    t0 = time.time()
    fh = open(csvinput)
    fh.readline() # Header

    #imageid,n,x,y,t

    #denom = np.ones(nsegments)

    inProj = Proj(init='epsg:3857')
    outProj = Proj(init='epsg:4326')

    nerrors = 0
    maxcount = 0
    
        
    #t0 = time.time()
    
    querycoords = []
    local_counts = []
    
    lats = []
    longs = []
    
    for i, line in enumerate(fh):
        #t1=time.time()
        #if i % 1000000:
        #    print(i,(t1-t0)/i)
        arr = line.split(',')
        count = int(arr[1])
        if not arr[2]:
            nerrors += 1
            continue
        longs.append(float(arr[2]))
        lats.append(float(arr[3]))
        local_counts.append(count)
        
    longs = np.array(longs)
    lats = np.array(lats)     
        
    longs, lats = transform(inProj,outProj,longs,lats)
    if count > maxcount: maxcount = count

    querycoords = np.hstack( (lats[:,None],longs[:,None]) ) #.append( (lat, lon) ) 
    _,artids = artpointstree.query(querycoords, k=1, n_jobs=-1)
    crossing_distance,corssing_index = crossing_pointstree.query(querycoords, k=1, n_jobs=-1)
    sids = artpoints[artids,2]
    
    
    nsegments = len(segments.keys())
    counts = np.zeros(nsegments)
    denom = np.zeros(nsegments)
    
    nintersections = crossing_points.shape[0]
    intersection_counts = np.zeros(nintersections)
    intersection_denom = np.zeros(nintersections)
    
    for idx in range(querycoords.shape[0]):
        if crossing_distance[idx] < intersection_delta:
            intersection_counts[int(corssing_index[idx])] += local_counts[idx]
            intersection_denom[int(corssing_index[idx])] += 1
        else:    
            counts[int(sids[idx])] += local_counts[idx]
            denom[int(sids[idx])] += 1

        
        #if len(querycoords) > 100000:
        #    process_point_set(querycoords,local_counts)
        #    querycoords = []
        #    local_counts = []
        

    #if len(querycoords) > 0:
    #    process_point_set(querycoords,local_counts)
    #    querycoords = []
    #    local_counts = []
    

        

    for i in range(nsegments):
        if denom[i] > 0:
            counts[i] /= denom[i]
            
    for i in range(nintersections):
        if intersection_denom[i] > 0:
            intersection_counts[i] /= intersection_denom[i]           
            
    debug(np.max(counts))
    debug('Max count:{}'.format(maxcount))
    warning('nerrors:{}'.format(nerrors))
    fh.close()
    debug('Points aggregated ({:.3f}s)'.format(time.time() - t0))
    return counts, intersection_counts



##########################################################
def render_matplotlib(nodeshash, segments, crossings, artpoints,crossing_points,
                      avgcounts,intersection_counts, outdir,
                      logplot=False,xlim=None,ylim=None):
    t0 = time.time() 
    fig, ax = plt.subplots(1,1, figsize=(4.5, 16))

    # Render nodes
    #nodes = get_nodes_coords_from_hash(nodeshash)
    #plt.scatter(nodes[:, 1], nodes[:, 0], c='blue', alpha=1, s=20)

    # Render artificial nodes
    #plt.scatter(artpoints[:, 1], artpoints[:, 0], c='blue', alpha=1, s=20)

    # Render segments
    segcolor = 'darkblue'
    i = 0
    maxvalue = np.log(25)
    
    angle = 0.62
    rot = np.array( [[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])

    lines = []
    values = []
    min_lats = []
    max_lats = []
    min_lons = []
    max_lons = []

    for wid, wnodes in segments.items():
        i += 1
        #r = lambda: random.randint(0,255)
        #if avgcounts == np.array([]):
        #    segcolor = '#%02X%02X%02X' % (r(),r(),r())
        #else:
        #    if avgcounts[wid] == 0: alpha = 0
        #    elif avgcounts[wid] > maxvalue: alpha = 1
        #    else: alpha = np.log(avgcounts[wid]) / maxvalue

        
        lats = []; lons = []
        for nodeid in wnodes:
            a, o = nodeshash[nodeid]
            lats.append(a)
            lons.append(o)
            
        rotated_vals = np.dot(np.column_stack([lons,lats]),rot) 

        lines.append( rotated_vals )
        values.append(avgcounts[wid])
        min_lats.append(np.min(rotated_vals[:,1]))
        max_lats.append(np.max(rotated_vals[:,1]))
        min_lons.append(np.min(rotated_vals[:,0]))
        max_lons.append(np.max(rotated_vals[:,0]))
        
    values = np.array(values)
    min_lats = np.array(min_lats)
    max_lats = np.array(max_lats)
    min_lons = np.array(min_lons)
    max_lons = np.array(max_lons)
    
    if xlim is None:
        min_ax = min( np.min(l[:,0]) for l in lines  )
        max_ax = max( np.max(l[:,0]) for l in lines  )
    else:
        min_ax = xlim[0]
        max_ax = xlim[1]
    
    if ylim is None:
        min_ay = min( np.min(l[:,1]) for l in lines  )
        max_ay = max( np.max(l[:,1]) for l in lines  )
    else:
        min_ay = ylim[0]
        max_ay = ylim[1]
        
    #values_to_mesure = np.logical_and(values>0 ,
    #                        np.logical_and(
    #                            np.logical_and(min_ay <= max_lats,max_ay >= min_lats),
    #                            np.logical_and(min_ax <= max_lons,max_ax >= min_lons)))
    values_to_mesure = values>0 
    
    ## Render crossings
    #crossingscoords = np.ndarray((len(crossings), 2))
    #for j, crossing in enumerate(crossings):
    #    crossingscoords[j, :] = np.array(nodeshash[crossing])
    ##plt.scatter(crossingscoords[:, 1], crossingscoords[:, 0], c='black')

    low_percentile = np.percentile(values[values_to_mesure],1.0)
    high_percentile = np.percentile(values[values_to_mesure],99.0)
    print(low_percentile,high_percentile)
    if logplot:
        cnorm = colors.LogNorm(low_percentile,high_percentile)
    else:
        cnorm = colors.Normalize(low_percentile,high_percentile)
        
    cmap = plt.get_cmap()
    cmap.set_under(cmap(0))
    cmap.set_bad(cmap(0))
    cmap.set_over(cmap(1))
    
    from matplotlib.collections import LineCollection
    line_segments = LineCollection(lines,
                                   #linewidths=(0.5, 1, 1.5, 2),
                                   linestyles='solid',
                                   norm=cnorm,
                                   cmap = cmap )
    line_segments.set_array(values)
    ax.add_collection(line_segments)
    axcb = fig.colorbar(line_segments,aspect=50,orientation='vertical')
    axcb.set_label('Relative Pedestrian Density',size=15)

    crossing_points_rot = np.dot(crossing_points,rot.T)
    plt.scatter(crossing_points_rot[:,1],crossing_points_rot[:,0],s=16,
                c=intersection_counts,cmap=cmap,norm=cnorm)
    
    
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle
    
    
    #patches = [Circle((crossing_points_rot[i,1],crossing_points_rot[i,0]),intersection_delta,
    #                  alpha=1,edgecolor='black',facecolor='none',fill=False,linewidth=4) 
    #                    for i in range(crossing_points_rot.shape[0])]
    

    #p = PatchCollection(patches,match_original=True)
    #ax.add_collection(p)
    
    #ax.axis('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.sci(line_segments)
    

        
    ax.set_xlim(min_ax, max_ax)
    ax.set_ylim(min_ay, max_ay)
    
    plt.savefig(os.path.join(outdir, 'test_new_map.png'))

    debug('Finished exporting to image ({:.3f}s)'.format(time.time() - t0))

##########################################################
def render_bokeh(nodeshash, ways, crossings, artpoints, counts):
    t0 = time.time() 
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

    debug('Not rendering count yet')
    debug('Finished rendering ({:.3f}s)'.format(time.time() - t0))
    show(p)  # open a browser

##########################################################
def compute_or_load(inputosm, countcsv, outdir):
    if os.path.exists(outdir):
        with open(os.path.join(outdir, 'nodeshash.pkl'), 'rb') as fh:
            nodeshash = pickle.load(fh)
        with open(os.path.join(outdir, 'segments.pkl'), 'rb') as fh:
            segments = pickle.load(fh)
        with open(os.path.join(outdir, 'crossings.pkl'), 'rb') as fh:
            crossings = pickle.load(fh)
        with open(os.path.join(outdir, 'artpoints.pkl'), 'rb') as fh:
            artpoints = pickle.load(fh)
        with open(os.path.join(outdir, 'crossing_points.pkl'), 'rb') as fh:
            crossing_points = pickle.load(fh)            
        with open(os.path.join(outdir, 'mycount.pkl'), 'rb') as fh:
            mycount = pickle.load(fh)
        with open(os.path.join(outdir, 'intersection_counts.pkl'), 'rb') as fh:
            intersection_counts = pickle.load(fh)
        debug('Successfully load files from {}'.format(outdir))
    else:
        os.mkdir(outdir)
        tree = ET.parse(inputosm)
        root = tree.getroot() # Tag osm
        ways, invways = parse_ways(root)
        nodeshash = parse_nodes(root, invways)
        ways, invways = filter_out_orphan_nodes(ways, invways, nodeshash)
        crossings = get_crossings(invways)
        segments, invsegments = get_segments(ways, crossings)
        artpoints = evenly_space_segments(segments, nodeshash)
        crossing_points = get_crossing_points(crossings,nodeshash)

        mycount,intersection_counts = get_count_by_segment(countcsv, segments,artpoints,crossing_points)
        tostore = {'nodeshash': nodeshash,
                   'segments': segments,
                   'crossings': crossings,
                   'artpoints': artpoints,
                   'crossing_points': crossing_points,
                   'mycount': mycount,
                   'intersection_counts': intersection_counts}

        for filename, item in tostore.items():
            fh = open(os.path.join(outdir, filename + '.pkl'),'wb') 
            pickle.dump(item, fh)
            fh.close()
    return nodeshash, segments, crossings, artpoints, crossing_points, mycount, intersection_counts


##########################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputosm', help='Input osm file')
    parser.add_argument('countcsv', help='Csv containing the count per segment')
    parser.add_argument('outdir', help='Output folder')
    parser.add_argument('--show', help='Show plot (requires display)', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)


    nodeshash, segments, crossings, artpoints,crossing_points, mycount,intersection_counts = \
        compute_or_load(args.inputosm, args.countcsv, args.outdir)

    xlim = None#(-83.9026,-83.8914)
    ylim = None#(-9.7359,-9.76126)

    render_matplotlib(nodeshash, segments, crossings, artpoints,crossing_points,mycount,
                      intersection_counts,args.outdir,logplot=True,xlim=xlim,ylim=ylim)