## This contains the functions for searching over streams and searching over locations

import pylab as pl
import sys
import datetime as dt
from datetime import timedelta as td
import numpy as np
from utils import *
import cPickle as pickle
import csv
#from sklearn.linear_model import Lasso, LassoCV, LinearRegression

print dt.datetime.now()
opts = parse_args()

# LOAD DATA
input = load_input(opts.input)

# PARSE ARGUMENTS

dates = unique(input['date'])

if opts.start_date == '':
    start_date = dates[0]
else:
    start_date = opts.start_date

if opts.end_date == '':
    end_date = dates[-1]
else:
    end_date = opts.end_date

period = (end_date - start_date).days + 1
daterange = [start_date+td(day) for day in range(period)] 

MAX_LAG = opts.lag
Y_LAG = opts.lag

streams = unique(input['type'])
#areas = unique(input['area'])
print "Time period:", start_date, "-", end_date

streams = streams[streams != opts.predict]
print "Searching the following streams", ' '.join(streams)
print "Predicting", opts.predict
n_streams = len(streams)

## UTILITY FUNCTION

def time_series(x,period=period,print_nonzero=False,lag=0,max_lag=MAX_LAG):
    r =  x['date'].groupby(x['date']).count().reindex(daterange).fillna(0) #.to_sparse()

    def rolling_mean(x,n):
        return x #stats.moments.ewma(x,span=n)

    if lag > 0:
        t = rolling_mean(r,7).shift(-1 * lag)[6:-1 * max_lag]
    else:
        t = rolling_mean(r,7)[6:-1 * max_lag]

    t = t.diff(7)[7:]

    #return t
    return t - t.mean()

## SEARCH FUNCTIONS

def exhaustive_stream_search(input,Y):
    max_subset = []
    max_value = -1

    for i, ss in enumerate(subsets(streams)):
        m = time_series(input[match_streams(input,ss)]).corr(Y)
        if m > max_value:
            max_value = m
            max_subset = ss
    
    return max_subset, max_value

def exhaustive_spatial_search(input,stream_s):
    max_subset = []
    max_value = -1
    tracts = unique(input['tract'])
    total = 2**len(tracts)

    X = input[match_streams(input, stream_s)]
    Y = input[match_streams(input, [opts.predict])]

    for i, ss in enumerate(subsets(tracts)):
        Xsubset = time_series(X[match_tracts(X,ss)])
        Ysubset = time_series(Y[match_tracts(Y,ss)],lag=Y_LAG)
        m = Xsubset.corr(Ysubset)
        if m > max_value:
            max_value = m
            max_subset = ss
    
    return max_subset, max_value

def lasso_stream_search(region,streams=streams,n_streams=n_streams):
    import numpy as np
    X = np.array([time_series(region[region['type'] == s]) for s in streams]).transpose()
    Y = time_series(region[region['type'] == opts.predict],lag=Y_LAG) 

    UseR = True
    if UseR:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri as npr
        import numpy as np
        from rpy2.robjects.packages import importr
        import rpy2
        rpy2.robjects.numpy2ri.activate()
        nls = importr('penalized')

        d = {'y':Y}
        predictors = '~'
        for i, s in enumerate(streams):
            d['x%d'%i] = time_series(region[region['type'] == s])
        predictors += '+'.join('x%d'%j for j in range(i))
        dataf = DataFrame(d)
        import os
        dataf.to_csv('/tmp/p%d.csv'%os.getpid())

    #    robjects.globalenv["dataf"] = dataf
        robjects.r("dataf<-read.csv('/tmp/p%d.csv')"%os.getpid())
        robjects.r('attach(dataf)')
        robjects.r('q <- optL1(y, ~%s, positive=TRUE,minlambda1=0,maxlambda1=1)'%predictors)
        #robjects.r('q <- optL1(y, ~%s, positive=TRUE,minlambda1=2,maxlambda1=2.08)'%predictors)
        coefs = robjects.r('coefficients(q$fullfit)')
        print "coefs", coefs
        coefs = list(robjects.r('names(coefficients(q$fullfit))'))
        print "coef names", coefs
        D = []
        for i in range(len(streams)):
            if "x%d"%i in coefs:
                D += [streams[i]]
        XforD = time_series(region[match_streams(region,D)])
        
        return sorted(D), XforD.corr(Y)

    else:
        lasso = LassoCV(eps=1e-5,n_alphas=1000,fit_intercept=False) 
        l = lasso.fit(X,Y)

        Xfit = Series(l.predict(X),index=Y.index) # do we need this? we don't report alpha anymore.

        print "LASSO COEF"
        for p, q in zip(streams,l.coef_):
            print "%s %.06f"%(p,q)
        D = streams[l.coef_ > 0]
        print "12345 alpha", l.alpha
        XforD = time_series(region[match_streams(region,D)])
        
        return sorted(D), XforD.corr(Y)

def iterative_stream_search(data, streams, max_iters=50):
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
#            print "\t***cor %.06f"%best_cor, "for streams", D
            finished = 0
        else:
#            print "\t   cor %.06f"%best_cor, "for streams", D
            finished = 1

#    print "++ rho=", best_cor
#    print "++ D", best_D
    return true_cor, best_D, i
        
def stream_search(region,streams=streams,n_streams=n_streams,streams_bestguess=streams, X_ts_precalculated=None, penalty=0):
    n_streams_bestguess = len(streams_bestguess)
    priority = np.zeros(n_streams)
    numer = np.zeros(n_streams)
    denom = np.zeros(n_streams)
    Q = np.zeros(n_streams)
    
    # calculate dependent variable in this region
    Y = time_series(region[region['type'] == opts.predict],lag=Y_LAG) 
    if np.sum(Y) == 0:
        X = time_series(region[match_streams(region,streams)])
        return streams, 0, X, Y, X_ts_precalculated
    
    if X_ts_precalculated == None:
        X_ts = {}
        precalculated = False
    else:
        X_ts = X_ts_precalculated
        precalculated = True
    for i, s1 in enumerate(streams):
        if not precalculated:
            try:
                X_ts[s1] = time_series(region[region['type'] == s1])
            except:
                X_ts[s1] = np.zeros(period)
        numer[i] = np.dot(X_ts[s1], Y)
        denom[i] = sqnorm(X_ts[s1])

    for i, s1 in enumerate(streams):
        Q[i] = 0
        for s2 in streams_bestguess:
            if s1 != s2:
                Q[i] += np.dot(X_ts[s1],X_ts[s2])
         
        if n_streams_bestguess > 0:
            Q[i] /= n_streams_bestguess
        
    approx_best = streams
    approx_cor = -1
    best_new_method = -1
    best_cor = -1
    
    # calculate correlation between each stream and opts.predict variable
    best = 0
    norm_y = pl.norm(Y)

    if n_streams_bestguess > 0:
        for D in range(1, len(streams)+1): 
            priority = numer / denom
             
            ii = range(n_streams)
            priority = nan_to_neg_inf(priority)
            ii = sorted(ii, key=lambda p: -1 * priority[p])

            cor = time_series(region[match_streams(region,streams[ii[0:D]])]).corr(Y) - penalty * D
            if cor > best_cor:
                best_cor = cor
                approx_best = streams[ii[0:D]]

            denom += Q
    else: # this gives the same result as if we had run the independence search
        priority = numer / denom
         
        ii = range(n_streams)
        priority = nan_to_neg_inf(priority)
        ii = sorted(ii, key=lambda p: -1 * priority[p])

        for D in range(1, len(streams)+1): 
            cor = time_series(region[match_streams(region,streams[ii[0:D]])]).corr(Y) - penalty * D
            if cor > best_cor:
                best_cor = cor
                approx_best = streams[ii[0:D]]

    X = time_series(region[match_streams(region,approx_best)])
    return sorted(approx_best), best_cor, X, Y, X_ts


def google_stream_search(region,streams=streams,n_streams=n_streams):
    priority = np.zeros(n_streams)
    
    # calculate dependent variable in this region
    Y = time_series(region[region['type'] == opts.predict],lag=Y_LAG) 
    if np.sum(Y) == 0:
        X = time_series(region[match_streams(region,streams)])
        return streams, 0, X, Y
    
    X_ts = {}

    for i, s1 in enumerate(streams):
        try:
            X_ts[s1] = time_series(region[region['type'] == s1])
        except:
            X_ts[s1] = np.zeros(period)
        priority[i] = Y.corr(X_ts[s1])

    approx_best = streams
    approx_cor = -1
    
    ii = range(n_streams)
    priority = nan_to_neg_inf(priority)
    ii = sorted(ii, key=lambda p: -1 * priority[p])

    best = -1
    approx_best = []
    for i in range(1,n_streams+1):
        cor = time_series(region[match_streams(region,streams[ii[0:i]])]).corr(Y)
        if cor > best:
            best = cor
            approx_best = streams[ii[0:i]]
            
    X = time_series(region[match_streams(region,approx_best)])
    return sorted(approx_best), best

def spatial_search(input, stream_s, tracts_bestguess, X_ts_precalculated=None, Y_ts_precalculated=None, X_tssq_precalculated=None, Y_tssq_precalculated=None):
    tracts = unique(input['tract'])
    n_tracts = len(tracts)
    #tracts_bestguess = tracts
    n_tracts_bestguess = len(tracts_bestguess)
    
    approx_best = []
    approx_cor = -1
    
    best = -1
    
    X = input[match_streams(input, stream_s)]
    Y = input[match_streams(input, [opts.predict])]
    
    Pxx = {}
    Pxy = np.zeros(n_tracts)
    Pyy = {}
    Q = 0
    R = 0
    
    numer = np.zeros(n_tracts)
    b_i = 0
    PX = 0
    PY = 0

    if X_ts_precalculated == None:
        X_ts = {}
        Y_ts = {}
        X_tssq = {}
        Y_tssq = {}
        precalculated = False
    else:
        X_ts = X_ts_precalculated
        Y_ts = Y_ts_precalculated
        X_tssq = X_tssq_precalculated
        Y_tssq = Y_tssq_precalculated
        precalculated = True

    if n_tracts <= 1:
        return tracts, 0, time_series(X), time_series(Y,lag=Y_LAG), X_ts_precalculated, Y_ts_precalculated, X_tssq_precalculated, Y_tssq_precalculated 

    for b_i, b in enumerate(tracts): 
        if not precalculated:
            X_ts[b] = time_series(X[X['tract'] == b])
            Y_ts[b] = time_series(Y[Y['tract'] == b],lag=Y_LAG)
            X_tssq[b] = sqnorm(X_ts[b])
            Y_tssq[b] = sqnorm(Y_ts[b])
        numer[b_i] = np.dot(X_ts[b],Y_ts[b])
        
        Pxx[b] = sqnorm(X_ts[b]) 
        Pyy[b] = sqnorm(Y_ts[b])
        PX += Pxx[b]
        PY += Pyy[b]

    for b in tracts:
        Pxx[b] = (PX - Pxx[b]) / (n_tracts-1.0)
        Pyy[b] = (PY - Pyy[b]) / (n_tracts-1.0)
     
    i = 0
    for b_i, b1 in enumerate(tracts):
        Pxy[b_i] = 0
        i += 1
        
        for b2 in tracts_bestguess:
            if b1 != b2:
                Pxy[b_i] += np.dot(X_ts[b1],Y_ts[b2])
                Q += np.dot(X_ts[b1], X_ts[b2])
                R += np.dot(Y_ts[b1], Y_ts[b2])
          
        try:
            Pxy[b_i] /= (n_tracts_bestguess - 1)
        except ZeroDivisionError:
            pass
    
    try:
        Q /= n_tracts_bestguess ** 2 - n_tracts_bestguess
        R /= n_tracts_bestguess ** 2 - n_tracts_bestguess
    except ZeroDivisionError:
        pass
    
    denom = np.zeros(n_tracts)
    for S in range(1, n_tracts+1): 
        denom = np.array([X_tssq[tract] * Y_tssq[tract] + X_tssq[tract] * (S-1.0) * Pyy[tract] / 2 + Y_tssq[tract] * (S-1.0) * Pxx[tract] / 2+ (S**2 - S + 0.0) * (X_tssq[tract] * R + Y_tssq[tract] * Q) + (S - 1.0)**2 * Q * R for i, tract in enumerate(tracts)])
        priority = numer / denom
        
        priority = nan_to_neg_inf(priority)
        ii = range(n_tracts)
        ii = sorted(ii, key=lambda p: -1 * priority[p])

        cor = time_series(X[X.tract.isin(tracts[ii[0:S]])]).corr(time_series(Y[Y.tract.isin(tracts[ii[0:S]])],lag=Y_LAG))
        if cor > best:
            best = cor
            approx_best = tracts[ii[0:S]]

        numer += Pxy

    return sorted(approx_best), best, time_series(X[X.tract.isin(approx_best)]), time_series(Y[Y.tract.isin(approx_best)],lag=Y_LAG), X_ts_precalculated, Y_ts_precalculated, X_tssq_precalculated, Y_tssq_precalculated 

def greedy_search(data, streams=streams):
    tracts= unique(data['tract'])
    Dopt = np.array([])
    Sopt = np.array([])
    beta = -1
    total_iters = 0

    for k in range(opts.restarts):
        #        print "restart #%d"%k
        best_D = random_subset(streams)
        best_S = tracts
        finished = 0
        i = 0

        best_cor = -1

        while(finished < 2 and i < 20):
            finished = 0
            i += 1
          
            cor, S = greedy_spatial(data, best_D)

            if cor > best_cor:
                best_S = S
                best_cor = cor
            else:
                finished = 1

            region = data[match_tracts(data,best_S)] 
            cor, D = greedy_streams(region,np.array(streams))

            if cor > best_cor:
                best_D = D
                best_cor = cor
            else:
                finished += 1

        total_iters += i

        if best_cor > beta:
            beta = best_cor
            Dopt = best_D
            Sopt = best_S

    return beta, list(Sopt), list(Dopt), (total_iters + 0.0) / opts.restarts

def greedy_streams(region,streams=streams):
    # calculate dependent variable in this region
    Y = time_series(region[region['type'] == opts.predict],lag=Y_LAG) 
    
    X_ts = {}
    n_streams = len(streams)

    for i in range(n_streams):
        try:
            X_ts[i] = time_series(region[region['type'] == streams[i]])
        except:
            X_ts[i] = Y * 0

    D = []
    Dopt = []
    X = np.zeros(len(Y))
    ropt = 0
    monotonic = True
    last_best = -1

    for ii in range(n_streams):
        r = np.zeros(n_streams)
        for j in range(n_streams):
            if not j in D:
                r[j] = Y.corr(X + X_ts[j]) 
                if ii == 0:
                    print "Y vs %s: %f"%(streams[j],r[j])

        r[np.isnan(r)] = 0

        max_i = np.argmax(r)
        X += X_ts[max_i]
        D.append(max_i)
        rstar = np.max(r)
        print "%d: r = %f, ropt = %f (%s)"%(ii,rstar,ropt,str(monotonic))
        if rstar > ropt:
            ropt = rstar
            Dopt = list(D)
#            if last_best != (ii - 1) and last_best != -1:
#                monotonic = False
            last_best = ii

    rtest = time_series(region[match_streams(region,streams[Dopt])]).corr(Y)
#    assert((ropt - rtest) < .00001)
    return ropt, list(streams[Dopt])

def greedy_locations(region,streams):
    tracts = unique(region['tract'])
    n_tracts = len(tracts)

    Xdata = region[match_streams(region, streams)]
    Ydata = region[match_streams(region, [opts.predict])]

    X_ts = {}
    Y_ts = {}

    for i in range(n_tracts):
        X_ts[i] = time_series(Xdata[Xdata['tract'] == tracts[i]])
        Y_ts[i] = time_series(Ydata[Ydata['tract'] == tracts[i]],lag=Y_LAG)

    S = []
    Sopt = []
    X = np.zeros(len(X_ts[0]))
    Y = np.zeros(len(X_ts[0]))
    ropt = 0

    for ii in range(n_tracts):
        r = np.zeros(n_tracts)
        for j in range(n_tracts):
            if not j in S:
                r[j] = (Y + Y_ts[j]).corr(X + X_ts[j]) 

        r[np.isnan(r)] = 0

        max_i = np.argmax(r)
        X += X_ts[max_i]
        Y += Y_ts[max_i]
        S.append(max_i)
        rstar = np.max(r)
        print "%d: r = %f, ropt = %f"%(ii,rstar,ropt)
        if rstar > ropt:
            ropt = rstar
            Sopt = list(S)

    rtest = time_series(Xdata[match_tracts(Xdata,tracts[Sopt])]).corr(time_series(Ydata[match_tracts(Ydata,tracts[Sopt])],lag=Y_LAG))
    #assert((ropt - rtest) < .00001)
    return ropt, list(tracts[Sopt])
