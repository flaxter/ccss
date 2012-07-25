#### This code searches for an optimal subset of streams, aggregated over a fixed set of locations

#### Ideas for exploratory data analysis:
#### 1. Try different sets of locations (very small vs. very large)

from ccss import *
from ccss import greedy_streams as search_streams

f = open(opts.output,"w")
out = csv.writer(f)
out.writerow("predict R Dopt S Xn Yn elapsed".split())

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
    tracts = get_tracts(input)
    for tract in tracts:
        data = select(input, tracts=[tract], fields=fields, start_date=start_date, end_date=end_date) #, input

        with mytimer:
            R, Dopt = search_streams(data, np.array(streams))

        Xn = np.sum(match_streams(data,Dopt))
        Yn = np.sum(match_streams(data,[opts.predict]))
        print "\n\n-------------------- In tract", tract
        print "Correlation = %.05f"%R
        print "for predicting", opts.predict, "with these leading indicators:", ', '.join(Dopt)
        print "# of events in X:", Xn
        print "# of events in Y:", Yn
        out.writerow([opts.predict, R, Dopt, tract, Xn, Yn, mytimer.elapsed])

        print "search complete, output written to %s"%(opts.output)

