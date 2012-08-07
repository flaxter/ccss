#### This code searches for an optimal subset of streams, aggregated over a fixed set of locations

#### Ideas for exploratory data analysis:
#### 1. Try different sets of locations (very small vs. very large)

from ccss import *
from ccss import greedy_streams as search_streams

f = open(opts.output,"w")
out = csv.writer(f)
out.writerow("predict R Dopt S Xn Yn elapsed".split())
out.writerow("predict R R_a R_b Dopt S Xn Yn elapsed".split())

matched = "all"

fields = 'type, tract, area, date'

if opts.areas != "all":
    print "Matching areas", opts.areas
    data = select(input, areas=opts.areas, fields=fields, start_date=start_date, end_date=end_date) #input[match_areas(input, opts.areas)]
    matched = opts.areas
elif opts.tracts != "all":
    print "Matching tracts", opts.tracts
    data = select(input, tracts=opts.tracts, fields=fields, start_date=start_date, end_date=end_date) #input[match_tracts(input, opts.tracts)]
    matched = opts.tracts
else:
    for matched in range(1,78):
        data = select(input, areas=[matched], fields=fields, start_date=start_date, end_date=end_date) #, input

        with mytimer:
            R_a, Dopt = search_streams(data, np.array(streams), daterange=TIME_PERIOD_A)


        X = match_streams(data,Dopt)
        Y = match_streams(data,[opts.predict]) 
        Xn = np.sum(X)
        Yn = np.sum(Y)
        X_ts = time_series(data[X], daterange=TIME_PERIOD_B)
        Y_ts = time_series(data[Y],lag=Y_LAG, daterange=TIME_PERIOD_B)
        R_b = X_ts.corr(Y_ts)
        X_ts = time_series(data[X], daterange=daterange)
        Y_ts = time_series(data[Y],lag=Y_LAG, daterange=daterange)
        R = X_ts.corr(Y_ts)
        print "\n\n-------------------- In area", matched
        print "Correlation = %.05f"%R
        print "for predicting", opts.predict, "with these leading indicators:", ', '.join(Dopt)
        print "# of events in X:", Xn
        print "# of events in Y:", Yn

        if R_a > opts.thresh and R_b > opts.thresh and R > opts.thresh: 
            print "Found a significant one, writing it to csv"
            out.writerow([opts.predict, R, R_a, R_b, Dopt, matched, Xn, Yn, mytimer.elapsed])

        print "search complete, output written to %s"%(opts.output)

