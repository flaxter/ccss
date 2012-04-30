# assign census tracts and areas to 311 calls 
import logging
logging.basicConfig(level=logging.DEBUG)

import csv
import pylab as pl
import numpy as np
import csv
from shapely.geometry import Point, Polygon

blocks = pl.csv2rec("blocks_to_coords.csv","r")

def nn(lat,long):
    return blocks['block'][np.argmin((blocks['lat'] - lat) ** 2 + (blocks['long'] - long) ** 2)]

import sys

if len(sys.argv) < 2:
    print "usage: %s input.csv"%sys.argv[0]
    sys.exit()

print "About to process..."
n_rows = sum(1 for line in open(sys.argv[1]))
print n_rows, "rows"
block_hash = {}

data = csv.reader(open(sys.argv[1],"r"))
#print len(blocks)
def match_block(lat,long,blocks=blocks):
    #print "match ",lat,long
    if (lat,long) in block_hash:
        return block_hash[(lat,long)]
    else:
	try:
            b = nn(lat,long)
            block_hash[(lat,long)] = b
            return b
        except:
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

print "Ready to start processing..."
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
               # print "didn't match", row
                pass
    except:
        pass
    i += 1
#    logging.debug("%f pct. complete"%((100.0 * i) / n_rows))
