from fetch_00 import *
import sys

def delete_last_row(filename):
    c =  "sed -i -e '$d' data/%s.csv"%filename
    print c
    os.system(c)

#for name in data:
#    delete_last_row(name)
     
if len(sys.argv) < 2:
    print "usage: %s output.csv"%sys.argv[0]
    sys.exit()

import csv
from pandas import *
first = True
for name in data:
    print name
    d = read_csv("data/%s.csv"%name,parse_dates=True)
    d = d.rename(columns={"Date":"date","Creation Date":"date", "CREATION DATE":"date","DATE SERVICE REQUEST WAS RECEIVED":"date", "IS THE BUILDING CURRENTLY VACANT OR OCCUPIED?":"type","LATITUDE":"latitude","LONGITUDE":"longitude","TYPE OF SERVICE REQUEST":"type","Type of Service Request":"type","Latitude":"latitude","Longitude":"longitude","Primary Type":"type", "X Coordinate":"X", "Y Coordinate":"Y", "X COORDINATE":"X", "Y COORDINATE":"Y"})
    print len(d), "rows"
    try:
        keep = np.array([not ('Dup' in q) for q in d.Status])
        d = d[keep]
    except AttributeError:
        pass

    print "Without duplicates:", len(d), "rows"

    d.date = [dstr[:10] for dstr in d.date]

    d.to_csv(sys.argv[1],mode="a",cols=['date','type','X','Y'],index=False,header=first)
    first = False

