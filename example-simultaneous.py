#### This code searches over subsets of streams and locations simultaneously

from ccss import *
from ccss import greedy_search as search


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
    R, Sopt, Dopt, iters = search(data, np.array(streams))

Xn = np.sum(match_streams(data,Dopt) * match_tracts(data,Sopt))
Yn = np.sum(match_streams(data,[opts.predict]) * match_tracts(data,Sopt))
print "\n\n-------------------- In area", matched, "with streams", streams
print "Correlation = %.05f"%R
print "for predicting", opts.predict, "with these leading indicators:", ', '.join(Dopt)
print "# of events in X:", Xn
print "# of events in Y:", Yn

f = open(opts.output,"w")
out = csv.writer(f)
out.writerow("predict streams region R Dopt Sopt Xn Yn elapsed".split())
out.writerow([opts.predict, streams, matched, R, Dopt, Sopt, Xn, Yn, mytimer.elapsed])

print "search complete, output written to %s"%(opts.output)
