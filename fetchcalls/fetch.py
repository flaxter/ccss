import os
import sys

data = {
"vacant": "7nii-7srd",
"sanitation": "me59-5fac",
"tree": "mab8-y9h3",
"potholes": "7as2-ds3y",
"rodent": "97t6-zrhs",
"grafitti": "hec5-y4x5",
"lights": "zuxi-7xem",
"garbagecarts": "9ksk-na4q",
#"all-crime": "ijzp-q8t2"
}

def s(x):
    print x
    os.system(x)

if len(sys.argv) < 2:
    print "usage: %s output.csv"%sys.argv[0]
    sys.exit()

for name in data.keys()[0:1]:
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

try:
    delete_row("garbagecarts")
except:
    print "failed to fix garbagecarts.csv"

try:
    delete_row("vacant")
except:
    print "failed to fix vacant.csv"
#delete_row("all-crime")

streams = [ "vacant-fixed", "sanitation", "tree", "potholes", "rodent", "grafitti", "lights", "garbagecarts-fixed"]

import csv
from pandas import *
first = True
for name in streams:
    try:
        d = read_csv("%s.csv"%name,parse_dates=True)
        d = d.rename(columns={"Creation Date":"date", "CREATION DATE":"date","DATE SERVICE REQUEST WAS RECEIVED":"date", "IS THE BUILDING CURRENTLY VACANT OR OCCUPIED?":"type","LATITUDE":"latitude","LONGITUDE":"longitude","TYPE OF SERVICE REQUEST":"type","Type of Service Request":"type","Latitude":"latitude","Longitude":"longitude"})
        d.to_csv(sys.argv[1],mode="a",cols=['date','type','latitude','longitude'],index=False,header=first)
        first = False
    except:
        print "failed on", name
        pass


