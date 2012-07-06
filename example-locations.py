# TODO: fix this...


#### This code searches for an optimal subset of locations, aggregated over a fixed set of streams

from ccss import *
from ccss import greedy_locations as search_locations

f = open(opts.output,"w")
out = csv.writer(f)

# Aggregate over these streams
if opts.streams != "all":
    print "Matching streams", opts.streams
    data = input[match_streams(input, opts.streams + [opts.predict])]
    streams = opts.streams
else:
    data = input

# Only consider these locations
if opts.areas != "all":
    print "Matching areas", opts.areas
    data = data[match_areas(data, opts.areas)]
    matched = opts.areas
elif opts.tracts != "all":
    print "Matching tracts", opts.tracts
    data = data[match_tracts(data, opts.tracts)]
    matched = opts.tracts

with mytimer:
    R, Sopt = search_locations(data, np.array(streams))

Xn = np.sum(match_tracts(data,Sopt))
Yn = np.sum(match_streams(data,[opts.predict]) * match_tracts(data,Sopt))
print "\n\n-------------------- In area", matched
print "Correlation = %.05f"%R
print "for predicting", opts.predict, "with these leading indicators:", ', '.join(streams)
print "in these tracts", Sopt
print "# of events in X:", Xn
print "# of events in Y:", Yn

out.writerow("predict R D Sopt Xn Yn elapsed".split())
out.writerow([opts.predict, R, streams, Sopt, Xn, Yn, mytimer.elapsed])

print "search complete, output written to %s"%(opts.output)
