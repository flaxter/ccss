import os
import sys

data = {
"vacant": "7nii-7srd",
"sanitation": "me59-5fac",
"tree_debris": "mab8-y9h3", # tree debris
"tree_trims":"uxic-zsuj",			# todo: need to add tree-trim 
"potholes": "7as2-ds3y",
"rodent": "97t6-zrhs",
"grafitti": "hec5-y4x5",
"lights": "zuxi-7xem",
"garbagecarts": "9ksk-na4q",
"all-crime": "ijzp-q8t2"
}

def s(x):
    print x
    os.system(x)

if len(sys.argv) < 2:
    print "usage: %s output.csv"%sys.argv[0]
    sys.exit()

# The average number of new reports per day is below:

#      Group.1         x
#1    graffiti 421.45175
#2      lights 207.25000
#3    potholes 213.22588
#4      rodent 177.40351
#5  sanitation  47.36842
#6 tree_debris  51.55281
#7  tree_trims  92.82743
#8      vacant  26.90728

for name in data.keys():
    s("wget http://data.cityofchicago.org/api/views/%s/rows.csv?accessType=DOWNLOAD"%data[name])
    s("mv rows.csv* %s.csv"%name)

def delete_row(filename):
    f = open("%s.csv"%filename, "r")
    f2 = open("%s-fixed.csv"%filename, "w")
    f2.write(f.next())
    f.next()
    for r in f:
        f2.write(r)
        
    f.close()
    f2.close()

#try:
#    delete_row("garbagecarts")
#except:
#    print "failed to fix garbagecarts.csv"

try:
    delete_row("vacant")
except:
    print "failed to fix vacant.csv"
#delete_row("all-crime")

delete_final_row() ?!?!

streams = [ "all-crime", "vacant-fixed", "sanitation", "tree_debris","tree_trims", "potholes", "rodent", "grafitti", "lights", "garbagecarts"]

import csv
from pandas import *
first = True
for name in streams:
    print name
    d = read_csv("%s.csv"%name,parse_dates=True)
    d = d.rename(columns={"Date":"date","Creation Date":"date", "CREATION DATE":"date","DATE SERVICE REQUEST WAS RECEIVED":"date", "IS THE BUILDING CURRENTLY VACANT OR OCCUPIED?":"type","LATITUDE":"latitude","LONGITUDE":"longitude","TYPE OF SERVICE REQUEST":"type","Type of Service Request":"type","Latitude":"latitude","Longitude":"longitude","Primary Type":"type", "X Coordinate":"X", "Y Coordinate":"Y", "X COORDINATE":"X", "Y COORDINATE":"Y"})
    print len(d), "rows"
    try:
        keep = np.array([not ('Dup' in q) for q in d.Status])
        d = d[keep]
    except AttributeError:
        pass

    print len(d), "rows"

    d.date = [dstr[:10] for dstr in d.date]

    d.to_csv(sys.argv[1],mode="a",cols=['date','type','X','Y'],index=False,header=first)
    first = False

