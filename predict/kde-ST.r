# Method 3 from Amrut's paper: use a kernel to smooth the violent crime data, plus other leading indicators, 
# from days -1 to days -28 to predict day 0

f = read.csv("geocoded.csv")
f$date = as.Date(f$date,format="%m/%d/%Y")
violent = subset(f,f$type=="VIOLENT")

MULTICORE = TRUE
if(MULTICORE) {
    library("doMC")
    registerDoMC()
}

library(sparr)
#library(spatstat)
library(spdep)
library(maptools)
library(sm)
library(fields)
#library(gpclib)
library("KernSmooth")
library(ROCR)
library(INLA)

beats = readShapePoly("../fetchcalls/shp/cpd_beats.shp")
city = readShapePoly("../fetchcalls/shp/City_Boundary.shp")
beats.centers = coordinates(beats)
city.bounds = as.owin(city)

beats.df = as.data.frame(beats)
beats.df = beats.df[,-3]
names(beats.df)[3] = "beat"
beats.df$beat = as.numeric(as.character(beats.df$beat))

if(!file.exists("../fetchcalls/shp/cpd_beats.graph")) {
    beats.nb = poly2nb(beats)
    nb2INLA(file="../fetchcalls/shp/cpd_beats.graph",beats.nb)
}

beats.to.id = data.frame(beats.df$beat, 1:nrow(beats.df))
names(beats.to.id) = c("beat","id")

beats.centers = as.data.frame(beats.centers)
beats.centers$beat = beats.df$beat
names(beats.centers) = c("X","Y","beat")

data = f[,c("X","Y", "type")]




gold = read.csv("input/gold-theta1.5-c5.csv")
gold$date = as.Date(gold$date)

GRID_SIZE = 400
kde.range = list(range(data$X),range(data$Y))
smoothed = bkde2D(data, 300, c(GRID_SIZE,GRID_SIZE), kde.range)

plot.kde = function(r) {
    image = im(t(r$fhat),xcol=r$x1,yrow=r$x2)
    plot(image)
}

xdatarange <- sort(rep(smoothed$x1,GRID_SIZE))
ydatarange <- rep(smoothed$x2,GRID_SIZE)
sp = SpatialPoints(data.frame(xdatarange,ydatarange))
smoothed.beats = over(sp, beats)
x = data.frame(xdatarange,ydatarange,smoothed.beats$BEAT_NUM,as.vector(t(smoothed$fhat)))
# debug:
# library(quilt)
# quilt.plot(x[,1],x[,2],x[,4],nx=GRID_SIZE,ny=GRID_SIZE)
names(x) = c("xdatarange","ydatarange","BEAT","fhat")
x$BEAT = as.numeric(as.character(x$BEAT))

ksmooth = function(data, x, streams, bw = 300, cluster.bw = 1000) {
    for(t in streams) {
        if(t == "violentclusters") {
            smoothed = bkde2D(subset(data,data$type==t), cluster.bw, c(GRID_SIZE,GRID_SIZE), kde.range)
        } else {
            smoothed = bkde2D(subset(data,data$type==t), 300, c(GRID_SIZE,GRID_SIZE), kde.range)
        }
        x[,t] = as.vector(t(smoothed$fhat))
    }
    return (x)
}

dates = unique(gold$date)
training.dates = dates[1:(length(dates)/2)]

#date.range = seq(as.Date("2011-01-01"), as.Date("2011-01-31"), 1)
date.range = seq(as.Date("2011-01-01"), dates[length(dates)], 1)

x.khat = data.frame()
for(day in 1:(length(date.range) - 27)) {
#x.bybeats = foreach(day=1:(length(date.range) - 27)) %dopar% {
    print(paste("day ",date.range[day]))
    month.start = date.range[day] 
    month.end = date.range[day+27] 

    ss = subset(f,(f$date >= month.start) & (f$date <= month.end))

    points = ss[,c("X","Y","type")]
    streams = unique(points$type)

    xhat = ksmooth(points, x, streams)
    xhat = subset(xhat, !is.na(xhat$BEAT))

    x.bybeat = aggregate(xhat[,5:ncol(xhat)], by=list(xhat$BEAT), FUN=sum)
    names(x.bybeat)[1] = "beat"
    x.bybeat$date = date.range[day+28]

    if(MULTICORE) {
        return(x.bybeat)
    } else {
        x.khat = rbind(x.khat, x.bybeat)
    }
}


if(MULTICORE) {
    for(r in 1:length(x.bybeats)) {
        x.khat = rbind(x.khat, x.bybeats[[r]])
    }
}

merged = merge(x.khat, gold, by=c("beat","date"), sort=FALSE)

training.data = merged
training.data$on[left.out] = NA
training.data = merge(training.data, beats.to.id, by="beat")

formula1 = on ~ VIOLENT + f(beat, model="iid") + f(id,model="besag",graph="../fetchcalls/shp/cpd_beats.graph") 
formula1 = on ~ VIOLENT + f(beat, model="iid") #+ f(id,model="besag",graph="../fetchcalls/shp/cpd_beats.graph") 
result2 = inla(formula1, family="binomial", data=training.data, Ntrials=1) #control.predictor=list(compute=TRUE), Ntrials=1)
predictions = inv.logit(result2$summary.fitted.values$mean)



### take a small subset of data to make fitting faster
q = ncol(merged) - 1
merged[,3:q] = apply(merged[,3:q], 2, function(x){ave(x,merged$beat, FUN=scale)})
merged = merge(merged, beats.to.id, by="beat", sort=FALSE)
left.out = !(merged$date %in% training.dates)

if(SMALL) {
    merged.ss = subset(merged,(merged$date >= as.Date("2011-04-01")) & (merged$date <= as.Date("2011-04-30")))
    left.out = (merged.ss$date <= as.Date("2011-04-30")) & (merged.ss$date > as.Date("2011-04-15"))
}

#training.data = merged.ss
#training.data$on[left.out] = NA

dtmp = merged[,c("on","VIOLENT","id","beat")]
dtmp$on[left.out] = NA

formula2 = on ~ VIOLENT + f(beat, model="iid") + f(id,model="besag",graph="../fetchcalls/shp/cpd_beats.graph") 
#formula2 = on ~ VIOLENT + f(id,model="besag",graph="../fetchcalls/shp/cpd_beats.graph") 
fit.inla3 = inla(formula2, family="binomial", data=dtmp, Ntrials=1, 
              # control.family = list(link = "logit"), 
               # control.compute=list(cpo=TRUE),
               control.predictor=list(compute=TRUE))
pred.inla = fit.inla3$summary.fitted.values$mean

fit.glm = glm(formula2, data=dtmp, family=binomial(link="logit"))
pred.glm = predict(fit.glm, merged, type="response") #subset(merged, left.out), type="response")
pdf("test.pdf")
plot(pred.glm, pred.inla[left.out])
dev.off()

#predictions.inla = inv.logit(result1$summary.fitted.values$mean)
pred = prediction(pred.glm[!left.out], merged$on[!left.out]) #$on[left.out]) #fit.glm$y) #merged.ss$on[!left.out])
auc = performance(pred,"auc")
print(paste("Original, !left.out", round(auc@y.values[[1]],4)))

pred = prediction(pred.glm[left.out], merged$on[left.out]) #merged.ss$on[!left.out])
auc = performance(pred,"auc")
print(paste("Original, left.out", round(auc@y.values[[1]],4)))

pred = prediction(pred.inla[!left.out], merged$on[!left.out]) #merged.ss$on[!left.out])
auc = performance(pred,"auc")
print(paste("INLA, !left.out", round(auc@y.values[[1]],4)))

pred = prediction(pred.inla[left.out], merged$on[left.out])
auc = performance(pred,"auc")
print(paste("INLA, left.out", round(auc@y.values[[1]],4)))

#predictions.inla = logit(result1$summary.fitted.values$mean)

n = nrow(training.data)
fitted.values.mean = numeric(n)
a = 0
b = 0
for(i in 1:n) {
    if (is.na(training.data$on[i])) {
        a = a + 1
        fitted.values.mean[i] = inla.expectation(function(x) exp(x)/(1 +exp(x)), result1$marginals.fitted.values[[i]])
    } else {
        b = b + 1
        fitted.values.mean[i] = result1$summary.fitted.values[i,"mean"]
    }
}
predictions.inla = fitted.values.mean


fit = glm(on ~ VIOLENT, data=subset(training.data,!left.out), family=binomial(link="logit"))
predictions = predict.glm(fit, merged.ss, type="response")


eval.fit(on ~ VIOLENT, "test.pdf","test", subset(training.data, !left.out), subset(orig,left.out))


    pred = prediction(predictions[!left.out], merged.ss$on[!left.out])
    perf = performance(pred,"tpr","fpr")
    auc = performance(pred,"auc")
    print(paste("Original, !left.out", round(auc@y.values[[1]],4)))

    pred = prediction(predictions[left.out], merged.ss$on[left.out])
    perf = performance(pred,"tpr","fpr")
    auc = performance(pred,"auc")
    print(paste("Original, left.out", round(auc@y.values[[1]],4)))

    pred = prediction(predictions.inla[!left.out], merged.ss$on[!left.out])
    perf = performance(pred,"tpr","fpr")
    auc = performance(pred,"auc")
    print(paste("INLA, !left.out", round(auc@y.values[[1]],4)))

    pred = prediction(predictions.inla[left.out], merged.ss$on[left.out])
    perf = performance(pred,"tpr","fpr")
    auc = performance(pred,"auc")
    print(paste("INLA, left.out", round(auc@y.values[[1]],4)))

#left.out = is.na(training.data$on)
pred = prediction(predictions[left.out], orig$on[left.out])

perf = performance(pred,"tpr","fpr")
auc = performance(pred,"auc")
print(round(auc@y.values[[1]],4))

pdf(filename)
plot(perf, xlim=c(0,.15), ylim=c(0,1), main=paste(title, "AUC=", round(auc@y.values[[1]],4)))

lines(c(0,1),c(0,1),col=2)
dev.off()
eval.fit = function(eq, filename, title, training.data, test.data) {
    fit = glm(eq, data=training.data, family=binomial(link="logit"))
    predictions = predict.glm(fit) #, type="response") #, test.data) #, type="response")

    pred = prediction(predictions, test.data$on)
    perf = performance(pred,"tpr","fpr")
    auc = performance(pred,"auc")
    print(paste(title, round(auc@y.values[[1]],4)))

    pdf(filename)
    plot(perf, xlim=c(0,.15), ylim=c(0,.75), main=paste(title, "AUC=", round(auc@y.values[[1]],4)))

    lines(c(0,1),c(0,1),col=2)
    dev.off()
}

