#!/usr/bin/env python3
"""Parse OSM data
"""

import argparse
import osmium
import shapely.wkb as wkblib


class WayLenHandler(osmium.SimpleHandler):
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.total = 0

    def way(self, w):
        wkbfab = osmium.geom.WKBFactory()
        wkb = wkbfab.create_linestring(w)
        line = wkblib.loads(wkb, hex=True)
        # Length is computed in WGS84 projection, which is practically meaningless.
        # Lets pretend we didn't notice, it is an example after all.
        self.total += line.length

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputosm', help='Input osm file')
    args = parser.parse_args()
    h = WayLenHandler()
    h.apply_file(args.inputosm, locations=True)
    print("Total length: %f" % h.total)

if __name__ == '__main__':
    main()

