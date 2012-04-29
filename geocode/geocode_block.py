# assign census tracts and areas to 311 calls 

import csv
import pylab as pl
import numpy as np
import csv
from shapely.geometry import Point, Polygon
from census_block import blocks
import sys

if len(sys.argv) < 2:
    print "usage: %s input.csv"%sys.argv[0]
    sys.exit()

data = csv.reader(open(sys.argv[1],"r"))
#print len(blocks)
def match_block(lat,long,blocks=blocks):
    #print "match ",lat,long
    for t in blocks.keys():
        if blocks[t].contains(Point(long,lat)):
            return t
    else:
        #print 'does not match'
        return -1

cols = data.next()
date_i = cols.index('date')
try:
    lat_i = cols.index('lat')
except:
    lat_i = cols.index('latitude')
try:
    long_i = cols.index('long')
except:
    long_i = cols.index('longitude')
cat_i = cols.index('type')

print "date,type,block"
cats = []
i = 0
for row in data:
    #print row
    try:
        date = row[date_i]
        cat = row[cat_i]
        lat = float(row[lat_i])
        long = float(row[long_i])
        #print date,cat,lat,long
        if cat != '' and date != '':
            try:
                block = match_block(lat,long)
                if(not block == -1): 
                    print "%s,%s,%s"%(date,cat.lower(),block)
            except:
                #print "didn't match", row
                pass
    except:
        pass
