from ccss import *
from iterative_search import search

print "precalculating centers of all tracts (this may take a little while)"
with mytimer:
    tract_centers = calculate_tract_centers(input)

print "completed in %d seconds"%mytimer.elapsed

out = csv.writer(open(opts.output,"w"))
out.writerow("tract R Dopt Sopt Xn Yn elapsed".split())

radius = .01

for tract in np.unique(input['tract']):
    data = input[match_tracts(input, nearby_tracts(tract_centers[tract],input,radius))]
    with mytimer:
        R, Sopt, Dopt, iters = search(data, np.array(streams))

    Xn = np.sum(match_streams(data,Dopt) * match_tracts(data,Sopt))
    Yn = np.sum(match_streams(data,[opts.predict]) * match_tracts(data,Sopt))
    print "\n\n-------------------- TRACT", tract
    print "Correlation = %.05f"%R
    print "for predicting", opts.predict, "with these leading indicators:", ' '.join(Dopt)
    print "# of events in X:", Xn
    print "# of events in Y:", Yn
    if np.abs(R) > .4:
        out.writerow([tract, R, Dopt, Sopt, Xn, Yn, mytimer.elapsed])

print "search complete, output written to %s"%(opts.output)
