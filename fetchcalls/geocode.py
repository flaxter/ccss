# assign census tracts and areas to 311 calls 

import csv
import pylab as pl
import numpy as np
import csv
from shapely.geometry import Point, Polygon
from census import tracts, tracts_to_area
import sys

if len(sys.argv) < 2:
    print "usage: %s input.csv"%sys.argv[0]
    sys.exit()

data = csv.reader(open(sys.argv[1],"r"))

def match_tract(lat,long,tracts=tracts):
    for t in tracts.keys():
        if tracts[t].contains(Point(long,lat)):
            return t
    else:
        return 'did not match'

cols = data.next()
date_i = cols.index('date')
lat_i = cols.index('latitude')
long_i = cols.index('longitude')
cat_i = cols.index('type')

print "date,type,area,tract,lat,long"
cats = []
i = 0
for row in data:
    try:
        date = row[date_i]
        cat = row[cat_i]
        lat = float(row[lat_i])
        long = float(row[long_i])

        if cat != '' and date != '':
            try:
                id = match_tract(lat,long)
                tract, area = tracts_to_area[id]
                print "%s,%s,%s,%s,%f,%f"%(date,cat.lower(),area,tract,lat,long)
            except:
                #print "didn't match", row
                pass
    except:
        pass
