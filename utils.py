import csv
import pylab as pl
import dateutil
import cPickle as pickle
import numpy as np
import logging
from pandas import *
logging.basicConfig(level=logging.DEBUG)

def cnt(x):
    return x.sum()

def corr(x,y):
    nx = pl.norm(x)
    ny = pl.norm(y)
    if nx == 0 or ny == 0:
        return 0
    else:
        return pl.dot(x,y) / (nx * ny)

def crosscorr(x,y,lag,max_lag=10):
    l = len(x)
    x = x[max_lag:l - max_lag]
    y = y[max_lag - lag:l - lag - max_lag]
    assert(len(x) == len(y))
    return pl.dot(x,y) / (pl.norm(x) * pl.norm(y))

def subset_data(data, x1, y1, x2, y2,area=-1):
    if area > 0:
        return (data['xcoordinate'] >= x1) * (data['xcoordinate'] <= x2) * (data['ycoordinate'] >= y1) * (data['xcoordinate'] <= y2) * (data['area'] == area)
    else:
        return (data['xcoordinate'] >= x1) * (data['xcoordinate'] <= x2) * (data['ycoordinate'] >= y1) * (data['xcoordinate'] <= y2) 

def center(l):
    return np.array(l) - np.mean(l)

def corr_center(x,y):
    return corr(center(x), center(y))

# source: http://wiki.python.org/moin/Powerful%20Python%20One-Liners
# modified from original to omit the empty set
subsets = lambda l: reduce(lambda z, x: z + [y + [x] for y in z], l, [[]])[1:]

def overlap(s1,s2):
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    s1 = set(s1)
    s2 = set(s2)
    return len(s1.intersection(s2)) / (1.0 * len(s1.union(s2)))

def match_areas(data, areas):
    return data['area'].isin(areas)

#    if areas[0] == -1:
#        return np.array([True] * len(data))
#    elif len(areas) > 0:
#        f = data['area'] == areas[0]
#        for d in areas[1:]:
#            f += data['area'] == d
#        return f
#    else:
#        return np.array([False] * len(data))

def match_streams(data, streams):
    return data['type'].isin(streams)

#    if len(streams) > 0:
#        f = data['type'] == streams[0]
#        for d in streams[1:]:
#            f += data['type'] == d
#        return f
#    else:
#        return np.array([False] * len(data))

def match_tracts(data, tracts):
    return data['tract'].isin(tracts)

#    if tracts == "all":
#        return np.array([True] * len(data))
#    if len(tracts) > 0:
#        f = data['tract'] == tracts[0]
#        for d in tracts[1:]:
#            f += data['tract'] == d
#        return f
#    else:
#        return np.array([False] * len(data))

def plot(input, streams, period, minX, maxX, minY, maxY, RECT_X, RECT_Y):
    pl.figure()

    for str in streams:
        f = input['type'] == str
        pl.plot(input[f]['xcoordinate'], input[f]['ycoordinate'], '.', label=str)

    pl.axis([minX, maxX, minY, maxY])
    ax = pl.subplot(111)
    ax.xaxis.set_major_locator(pl.MultipleLocator(RECT_X))
    ax.yaxis.set_major_locator(pl.MultipleLocator(RECT_Y))
    ax.xaxis.grid(True,'major')
    ax.yaxis.grid(True,'major')
    pl.legend(loc='lower left')
    pl.savefig("current-fig.png")

def sqnorm(x):
    return np.dot(x,x)

def parse_args():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", help="name of input file in .csv format", default="input/sample.csv")
    parser.add_option("-o", "--output", dest="output", help="name of output file", default="results/default-output.csv")
    parser.add_option("--output_prefix", dest="output_prefix", help="name of output prefix", default="")
    parser.add_option("-s", "--streams", dest="streams", help="independent variable stream (use 'all' to try all subsets)", default="all")
    parser.add_option("-p", "--predict", dest="predict", help="dependent variable stream (what do you want to predict?)", default="graffiti removal")
    parser.add_option("-a", "--area", dest="areas", help="search over which areas? (use 'all' for all, or put a list in quotes)", default="all")
    parser.add_option("-t", "--tracts", dest="tracts", help="restrict data to which tracts? (put a list in quotes)", default="all")
    parser.add_option("--test", dest="test", help="script specific", default="")
    parser.add_option("--noniterative", dest="iterative", help="for ADP-iterative or not?", default=True, action="store_false")
    parser.add_option("--restarts", dest="restarts", help="number of random restarts", default=10, type=int)
    parser.add_option("--image", dest="image", help="image output file for plotting", default="output.png")
    parser.add_option("--skip", dest="skip_filter", action="store_true", help="for debugging only", default=False)
    parser.add_option("--exhaustive", dest="exhaustive", help="pickled dict with exhaustive results, indexed by area", default='')
    parser.add_option("--window", dest="window", help="for plots, collapse by week (window=7) or month (window=30)", default=7,type=int)
    parser.add_option("--aggregate", dest="aggregate", help="for exhaustive search only, calculate F(S|D) (use --aggregate D) or F(D|S) (use --aggregate S)", default='')
    parser.add_option("--startdate", dest="start_date", help="MM/DD/YYYY", default='')
    parser.add_option("--enddate", dest="end_date", help="MM/DD/YYYY", default='')
    parser.add_option("--lag", dest="lag", default=7, type=int)
    parser.add_option("--dpenalty", dest="dpenalty", default=0, type=float)
    parser.add_option("--process", dest="process")
    parser.add_option("--row", dest="row", type=int)

    opts = parser.parse_args()[0]
    if opts.start_date != '':
        from dateutil import parser as p
        opts.start_date = p.parse(opts.start_date).date()
    if opts.end_date != '':
        from dateutil import parser as p
        opts.end_date = p.parse(opts.end_date).date()
    if opts.areas != "all":
	if "," in opts.areas:
            opts.areas = [float(s) for s in opts.areas.strip().strip('[]').split(',')]
	else:
            opts.areas = [float(s) for s in opts.areas.strip().strip('[]').split()]
    if opts.tracts != "all":
	if "," in opts.tracts:
            opts.tracts = [float(s) for s in opts.tracts.strip().strip('[]').split(',')]
	else:
            opts.tracts = [float(s) for s in opts.tracts.strip().strip('[]').split()]
    if opts.streams != "all" and opts.streams != "best":
        opts.streams = [s.strip().strip("'") for s in opts.streams.strip().strip('[]').split(',')]

    return opts

def center(l):
    return np.array(l) - np.mean(l)

class TooSparse(Exception):
    def __init__(self,value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def load_input(filename):
    f = filename + ".p"
    try:
        logging.debug("Trying to load %s"%f)
        input = pickle.load(open(f,"rb"))
    except:
        logging.debug("Loading %s instead"%filename)
        input = read_csv(filename,converters={'date':lambda d: dateutil.parser.parse(d).date()})
	try:
            pickle.dump(input, open(f, "wb"))
            logging.debug("Saving %s"%f)
	except:
            pass
    return input

import random
rb = lambda: {0:False,1:True}[random.randint(0,1)]
def random_subset(s):
    mask = np.array([rb() for n in range(len(s))])
    while(np.sum(mask) == 0): # non-empty subsets only, please
        mask = np.array([rb() for n in range(len(s))])
    return s[mask]

import datetime
default_daterange = [datetime.date(2011,1,1)+datetime.timedelta(day) for day in range(365)]

def time_series(x,max_lag,daterange=default_daterange,print_nonzero=False,lag=0):
    print "using time_series in utils.py"
    r =  x['date'].groupby(x['date']).count().reindex(daterange).fillna(0) #.to_sparse()
    if lag > 0:
        t = rolling_mean(r,7).shift(-1 * lag)[6:-1 * max_lag]
    else:
        t = rolling_mean(r,7)[6:-1 * max_lag]

    return t - t.mean()

def xy(input,streams,tracts,areas,predict,lag):
    mask = np.array([True] * len(input))
    if tracts != "all":
        mask = mask & (input['tract'].isin(tracts))

    if areas != "all":
        mask = mask & (input['area'].isin(area))
    
    Y = time_series(input[mask & (input['type'] == predict)],max_lag=lag,lag=lag)
    if streams != "all":
        mask = mask & (input['type'].isin(streams))
    
    X = time_series(input[mask],max_lag=lag)
    
    return X, Y

import time
class Timer():
    def __enter__(self): self.start = time.time()
    def __exit__(self, *args): self.elapsed = time.time() - self.start

mytimer = Timer()

def unique(x):
    return np.unique(np.array(x))

def nan_to_neg_inf(x):
    x = np.array(x)
    x[np.isnan(x)] = -np.inf
    return x

def nearby_tracts(coord, input, radius=.01):
    lat, long = coord
    distances = np.sqrt((input['lat'] - lat) ** 2 + (input['long'] - long) ** 2)
    return np.array(np.unique(input[distances < radius]['tract']))

def calculate_tract_centers(input=0):
    tract_centers = {}
    c = csv.reader(open("geocode/tracts_to_coords.csv","r"))
    col = c.next()
    for row in c:
        tract_centers[float(row[0])] = (float(row[1]),float(row[2]))
    return tract_centers

   # for t in np.unique(input['tract']):
   #     x = input[input['tract'] == t]
   #     lat = np.median(x['lat'])
   #     long = np.median(x['long'])
   #     tract_centers[t] = (lat,long)
#
#    return tract_centers
