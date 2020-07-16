## Studying OSM
Install
```
conda create --name carmeravis bokeh pandas pillow datashader matplotlib ipython pyproj rtree fiona descartes geojson astropy
pip install mpl-scatter-density
```


The script `src/renderseq.py` generates a blurred scatter of the points, colored by the counts.

The script `src/renderstreets.py` plots the counts colored by streets.

```
python src/renderstreets.py  data/data.osm   data/data.csv /tmp/rendered/
```
