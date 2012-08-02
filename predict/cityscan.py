import csv
from pandas import *
import utils
from dateutil import parser
from datetime import timedelta as td
import datetime
from optparse import OptionParser
import csv

AREAL = "beat"
end_date = datetime.date(2012, 4, 30) 
outf = "out.csv"

if __name__  == "__main__":
    op = OptionParser()
    op.add_option("--areal", dest="areal",help="areal unit to use (default is beat)", default="beat")
    op.add_option("--output", dest="output",help="output file use (default is out.csv)", default="out.csv")
    op.add_option("--input", dest="input",help="input file (default is geocoded.csv)", default="geocoded.csv")
    op.add_option("--stream", dest="stream",help="stream (default is VIOLENT)", default="VIOLENT")
    op.add_option("--date", dest="date",help="date on which to detect anomalies (assumes there are 28 days previous to this date in the data), use format YYYY-MM-DD (default is 2012-04-30)",default="2012-04-30")

    opts = op.parse_args()[0]
    outf = opts.output
    AREAL = opts.areal
    end_date = parser.parse(opts.date).date()

vc = csv.reader(open(opts.input,"r"))
cols = vc.next()
start_date = end_date - td(28) 

date_i = cols.index('date')
stream_i = cols.index('type')
areal_i = cols.index(AREAL)

n_dates = (end_date - start_date).days + 1

fo = open(outf,"w")
out = csv.writer(fo)
out.writerow(['current',AREAL,'type'])

for row in vc:
    date, stream, beat = row[date_i], row[stream_i], row[areal_i]
    date = parser.parse(date).date()
    
    if date <= end_date and date >= start_date and beat != 'NA' and stream == opts.stream:
        print "MATCHED! date", date, "beat", beat, "stream", stream
        out.writerow([int(date==end_date), beat, stream])
    else:
        print "DID NOT MATCH: date", date, "beat", beat, "stream", stream

fo.flush()
