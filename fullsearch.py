from ccss import *
import random

def search_S(data, tracts, D, max_iters=50):
    beta = -1
    Sopt = np.array([])
    cache1 = None
    cache2 = None
    cache3 = None
    cache4 = None

    best_S = tracts
    S = best_S
    finished = 0
    i = 0

    best_cor = -1
    last_S = np.array([])

    while(finished < 1 and i < max_iters):
        finished = 0
        i += 1
      
        S, cor, X, Y, cache1, cache2, cache3, cache4 = spatial_search(data, D, best_S, cache1, cache2, cache3, cache4) 

        if list(S) == list(last_S):
            finished = 1
        last_S = S
        
        if cor > best_cor:
            best_S = S
            best_cor = cor
            finished = 0
            #print "\t***cor %.06f"%best_cor, "for tracts", S
        else:
            finished = 1
            #print "\t   cor %.06f"%best_cor, "for tracts", S

    if best_cor > beta:
        beta = best_cor
        Sopt = best_S

    #print "++ rho=", beta
    #print "++ D", D
    #print "++ S", Sopt
    return beta, Sopt

def search_D(data, streams, max_iters=50):
    q = None

    D = np.array([])
    best_D = np.array([])
    finished = 0
    i = 0

    best_cor = -1
    true_cor = -1

    while(finished < 1 and i < max_iters):
        finished = 0
        i += 1
      
        D, cor, X, Y, q = stream_search(data, np.array(streams), len(streams), D, q) 
        
        if cor > best_cor:
            best_D = D
            best_cor = cor
            true_cor = X.corr(Y)
            #print "\t***cor %.06f"%best_cor, "for streams", D
            finished = 0
        else:
            #print "\t   cor %.06f"%best_cor, "for streams", D
            finished = 1

#    print "++ rho=", best_cor
#    print "++ D", best_D
    return true_cor, best_D, i


def search(data, streams=streams):
    tracts= unique(data['tract'])
    Dopt = np.array([])
    Sopt = np.array([])
    beta = -1
    total_iters = 0

    for k in range(opts.restarts):
        print "restart #%d"%k
        best_D = random_subset(streams)
        best_S = tracts
        print "D_0", best_D
        print "S_0", list(best_S)
        finished = 0
        i = 0

        best_cor = -1

        while(finished < 2 and i < 20):
            finished = 0
            i += 1
          
            cor, S = search_S(data, np.array(best_S), best_D)

            if cor > best_cor:
                best_S = S
                best_cor = cor
                print "\t*** %.06f"%best_cor, "for tracts", S
            else:
                finished = 1
                print "\txxx %.06f"%cor, "for tracts", S

            region = data[match_tracts(data,best_S)] 
            cor, D, d_i = search_D(region,np.array(streams))

            if cor > best_cor:
                best_D = D
                best_cor = cor
                print "\t*** %.06f"%best_cor, "for streams", D
            else:
                finished += 1
                print "\txxx %.06f"%cor, "for streams", D

        total_iters += i

        if best_cor > beta:
            beta = best_cor
            print "here, updating Dopt = ", Dopt, "Sopt = ", Sopt
            Dopt = best_D
            Sopt = best_S

    print "++ beta=", beta
    print "++ D", Dopt
    print "++ S", Sopt
    return beta, list(Sopt), list(Dopt), (total_iters + 0.0) / opts.restarts

