#### This code searches for an optimal subset of streams, aggregated over a fixed set of locations

#### Ideas for exploratory data analysis: 
#### 1. Try different sets of locations (very small vs. very large)

from ccss import *
from ccss import greedy_streams as search_streams

f = open(opts.output,"w")
out = csv.writer(f)
out.writerow("predict R Dopt S Xn Yn elapsed".split())

matched = "all"

if opts.areas != "all":
    print "Matching areas", opts.areas
    data = input[match_areas(input, opts.areas)]
    matched = opts.areas
elif opts.tracts != "all":
    print "Matching tracts", opts.tracts
    data = input[match_tracts(input, opts.tracts)]
    matched = opts.tracts
else:
    data = input

with mytimer:
    R, Dopt = search_streams(data, np.array(streams))

Xn = np.sum(match_streams(data,Dopt))
Yn = np.sum(match_streams(data,[opts.predict]))
print "\n\n-------------------- In area", matched
print "Correlation = %.05f"%R
print "for predicting", opts.predict, "with these leading indicators:", ', '.join(Dopt)
print "# of events in X:", Xn
print "# of events in Y:", Yn
out.writerow([opts.predict, R, Dopt, matched, Xn, Yn, mytimer.elapsed])

print "search complete, output written to %s"%(opts.output)
