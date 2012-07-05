#### This code searches over subsets of streams and locations simultaneously
#### The search happens centered on every tract in the city, with a radius (defined below) of .01

#### Ideas for exploratory data analysis: 
#### 1. Tweak this radius
#### 2. Leave out 50% of data (create a training dataset and a testing dataset), run this on
####    the training data set and see whether any of the correlations you find are still significant
####    for the test data set.

RADIUS = .01

from ccss import *
from ccss import greedy_search as search

print "precalculating centers of all tracts (this may take a little while)"
with mytimer:
    tract_centers = calculate_tract_centers(input)

print "completed in %d seconds"%mytimer.elapsed

f = open(opts.output,"w")
out = csv.writer(f)
out.writerow("predict tract R Dopt Sopt Xn Yn elapsed".split())


n_tracts = len(np.unique(input['tract'])) * 1.0
for i, tract in enumerate(np.unique(input['tract'])):
    data = input[match_tracts(input, nearby_tracts(tract_centers[tract],input,RADIUS))]
    with mytimer:
        R, Sopt, Dopt, iters = search(data, np.array(streams))

    Xn = np.sum(match_streams(data,Dopt) * match_tracts(data,Sopt))
    Yn = np.sum(match_streams(data,[opts.predict]) * match_tracts(data,Sopt))
    print "\n\n-------------------- TRACT", tract
    print "Correlation = %.05f"%R
    print "for predicting", opts.predict, "with these leading indicators:", ' '.join(Dopt)
    print "# of events in X:", Xn
    print "# of events in Y:", Yn
    if np.abs(R) > .3:
        out.writerow([opts.predict, tract, R, Dopt, Sopt, Xn, Yn, mytimer.elapsed])
    f.flush()
    print "%d %% done", i / n_tracts

print "search complete, output written to %s"%(opts.output)
