import csv
import sys
if len(sys.argv) < 2:
    print "usage: %s input.csv"%sys.argv[0]
    sys.exit()

data = csv.reader(open(sys.argv[1],"r"))
cols = data.next()
#date,type,area,tract,lat,long
date_i = cols.index('date')
type_i = cols.index('type')
area_i = cols.index('area')
tract_i = cols.index('tract')
lat_i = cols.index('lat')
long_i = cols.index('long')


for row in data:
	date = row[date_i]
	data_type = row[type_i]
	lati = float(row[lat_i])
	longi = float(row[long_i])
	print "%s,%s,%f,%f"%(date,data_type,lati,longi)


