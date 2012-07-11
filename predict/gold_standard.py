import sqlite3 as sql
from pandas import *
import utils
from dateutil import parser
from datetime import timedelta as td
import datetime

db = utils.load("input.db")

# get a list of areal units
areals = utils.query(db,"SELECT DISTINCT beat FROM data")
n_areals = len(areals)
print "Number of areal units:", n_areals

#dates = utils.query(db,"SELECT DISTINCT date FROM data WHERE date between '2011-01-01' and '2012-06-30'")
#dates = [parser.parse(d).date() for d in dates]


vc = read_csv("violent-by-beat.csv",converters={'date':lambda d: parser.parse(d).date()})

# get a list of dates
start_date = min(vc.date)
end_date = datetime.date(2012, 4, 30) #max(vc.date)
print "Date range:", start_date, "-", end_date

n_dates = (end_date - start_date).days + 1
Braw = {}
Craw = {}
Coverall = np.zeros(n_dates)

for date,beat in vc.values:
    if date <= end_date and date >= start_date: 
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
chk = 0.0
THETA_THRESH = 2
C_THRESH = 5
n_on = {}

for beat in Craw.keys():
    C[beat] = rolling_sum(Craw[beat],7)[6:]

    B[beat] = (total_by_beat[beat] / Ctotal) * rolling_total
    chk += total_by_beat[beat] / Ctotal
    theta[beat] = (C[beat] - B[beat]) / np.sqrt(B[beat])

    gold[beat] = (theta[beat] > THETA_THRESH) * (C[beat] > C_THRESH)
    n_on[beat] = sum(gold[beat])


pl.plot(range(len(gold[beat])),Craw[beat][6:])
pl.plot(range(len(gold[beat])),gold[beat]*8, '.', markersize=25)
pl.show()

