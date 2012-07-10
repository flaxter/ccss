#### This code searches over subsets of streams and locations simultaneously
#### The search happens centered on every tract in the city, with a radius (defined below) of .01

RADIUS = .01

from ccss import *
from ccss import greedy_search as search

print "precalculating centers of all tracts (this may take a little while)"
with mytimer:
    tract_centers = calculate_tract_centers(input)

print "completed in %d seconds"%mytimer.elapsed

f = open(opts.output,"w")
out = csv.writer(f)
out.writerow("predict tract R R_a R_b Dopt Sopt Xn Yn elapsed".split())

n_tracts = len(np.unique(input['tract'])) * 1.0
for i, tract in enumerate(np.unique(input['tract'])):
    data = input[match_tracts(input, nearby_tracts(tract_centers[tract],input,RADIUS))]
    with mytimer:
        R_a, Sopt, Dopt, iters = search(data, np.array(streams), daterange=TIME_PERIOD_A)

    X = match_streams(data,Dopt) * match_tracts(data,Sopt)
    Y = match_streams(data,[opts.predict]) * match_tracts(data,Sopt)
    Xn = np.sum(X)
    Yn = np.sum(Y)
    X_ts = time_series(data[X], daterange=TIME_PERIOD_B)
    Y_ts = time_series(data[Y],lag=Y_LAG, daterange=TIME_PERIOD_B)
    R_b = X_ts.corr(Y_ts)
    X_ts = time_series(data[X], daterange=daterange)
    Y_ts = time_series(data[Y],lag=Y_LAG, daterange=daterange)
    R = X_ts.corr(Y_ts)

    print "\n\n-------------------- TRACT", tract
    print "Correlation = %.05f (%.05f for first half of year, %.05f for second half of year)"%(R,R_a,R_b)
    print "for predicting", opts.predict, "with these leading indicators:", ', '.join(Dopt)
    print "# of events in X:", Xn
    print "# of events in Y:", Yn
    #if R_a > .3 and R_b > .3 and R > .3:
    out.writerow([opts.predict, tract, R, R_b, Dopt, Sopt, Xn, Yn, mytimer.elapsed])
    f.flush()
    print "%d %% done"%(i / n_tracts * 100)

print "search complete, output written to %s"%(opts.output)
