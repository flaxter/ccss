import csv
from pandas import *
import utils
from dateutil import parser
from datetime import timedelta as td
import datetime
from optparse import OptionParser
import csv


THETA_THRESH = 2
C_THRESH = 5
AREAL = "beat"
start_date = datetime.date(2011, 1, 1) #min(vc.date)
end_date = datetime.date(2012, 4, 30) #max(vc.date)
outf = "out.csv"

if __name__  == "__main__":
    op = OptionParser()
    op.add_option("--theta",dest="theta",help="threshold for theta (default is 2)", default=2, type=float)
    op.add_option("-c", dest="c",help="threshold for counts c (default is 5)", default=5, type=float)
    op.add_option("--areal", dest="areal",help="areal unit to use (default is beat)", default="beat")
    op.add_option("--output", dest="output",help="output file use (default is out.csv)", default="out.csv")
    op.add_option("--input", dest="input",help="input file (default is geocoded.csv)", default="geocoded.csv")
    op.add_option("--stream", dest="stream",help="stream type to predict (default is VIOLENT)", default="VIOLENT")
    op.add_option("--start", dest="start",help="start date, use format YYYY-MM-DD (default is 2011-01-01)",default="2011-01-01")
    op.add_option("--stop", dest="end",help="stop date, use format YYYY-MM-DD (default is 2012-03-31)",default="2012-03-31")

    opts = op.parse_args()[0]

    outf = opts.output
    THETA_THRESH = opts.theta
    C_THRESH = opts.c
    AREAL = opts.areal
    start_date = parser.parse(opts.start).date()
    end_date = parser.parse(opts.end).date()

vc = csv.reader(open(opts.input,"r"))
cols = vc.next()

date_i = cols.index('date')
stream_i = cols.index('type')
areal_i = cols.index(AREAL)

n_dates = (end_date - start_date).days + 1
Braw = {}
Craw = {}
Coverall = np.zeros(n_dates)

for row in vc:
    date, stream, beat = row[date_i], row[stream_i], row[areal_i]
    date = parser.parse(date).date()
    
    if date <= end_date and date >= start_date and beat != 'NA' and stream == opts.stream:
        t0 = (date - start_date).days
        if not beat in Craw:
            Craw[beat] = np.zeros(n_dates)
        
        Craw[beat][t0] += 1
        Coverall[t0] += 1

total_by_beat = {}
for beat in Craw.keys():
    total_by_beat[beat] = np.sum(Craw[beat])

Ctotal = np.sum(total_by_beat.values())

C = {}
B = {}
gold = {}
theta = {}

rolling_total = rolling_sum(Coverall,7)[6:]
n_on = {}
n_on_tot = 0

for beat in Craw.keys():
    C[beat] = rolling_sum(Craw[beat],7)[6:]

    B[beat] = (total_by_beat[beat] / Ctotal) * rolling_total
    theta[beat] = (C[beat] - B[beat]) / np.sqrt(B[beat])

    gold[beat] = (theta[beat] > THETA_THRESH) * (C[beat] > C_THRESH)
    n_on[beat] = sum(gold[beat])
    n_on_tot += sum(gold[beat])

#import pylab as pl
#pl.plot(range(len(gold[beat])),Craw[beat][6:])
#pl.plot(range(len(gold[beat])),gold[beat]*8, '.', markersize=25)
#pl.show()
print "Total number of areal units that are on over time", n_on_tot

fo = open(outf,"w")
out = csv.writer(fo)
out.writerow([AREAL, 'date', 'on'])

start_date = start_date + td(6)
n_dates = (end_date - start_date).days + 1

for beat in gold.keys():
    for i in range(n_dates):
        out.writerow([beat, start_date + td(i), gold[beat][i]])

fo.flush()
