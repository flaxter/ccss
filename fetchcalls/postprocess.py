# do some cleanup on the data, and select the dates of interest

import sys
import csv
import pylab as pl
import numpy as np
import csv
from dateutil import parser
import datetime

if len(sys.argv) < 2:
    print "usage: %s input.csv"%sys.argv[0]
    sys.exit()

data = csv.reader(open(sys.argv[1],"r"))
cols = data.next()

print ",".join(cols)
types = set([])
old = ''
s1 = [ "vacant", "sanitation", "tree_debris","tree_trims", "potholes", "rodent", "graffiti", "lights", "garbagecarts", "lights"]
s2 = [s.lower() for s in ['Vacant', 'Sanitation Code Violation', 'Tree Debris',"Tree Trim", 'Pot Hole in Street', 'Rodent Baiting/Rat Complaint', 'Graffiti Removal', 'Street Light - All/Out', 'Garbage Cart Black Maintenance', 'Street Lights - All/Out']]

streams = dict(zip(s2,s1))
dates = {}

i = 0
for row in data:
    i += 1
    date = row[0]
    dt = parser.parse(row[0])
    type = row[1]
    area = row[2]
    tract = row[3]

    if type in s2 and dt >= datetime.datetime(2011,1,1) and dt <= datetime.datetime(2012,3,31):
        row[1] = streams[row[1]]
        print ",".join(row)

