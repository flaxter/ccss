# Method 2 from Amrut's paper: use a kernel to smooth the violent crime data from days -1 to days -28 to predict
# day 0

f = read.csv("violent.csv")
f$date = as.Date(f$date,format="%m/%d/%Y")

#library("doMC")
#registerDoMC()

library(sparr)
#library(spatstat)
library(spdep)
library(maptools)
library(sm)
#library(gpclib)
library("KernSmooth")
library(ROCR)


beats = readShapePoly("../fetchcalls/shp/cpd_beats.shp")
city = readShapePoly("../fetchcalls/shp/City_Boundary.shp")
beats.centers = coordinates(beats)
city.bounds = as.owin(city)

beats.df = as.data.frame(beats)
beats.df = beats.df[,-3]
names(beats.df)[3] = "beat"
beats.df$beat = as.numeric(as.character(beats.df$beat))

data = f[,c("X","Y")]
gold = read.csv("input/gold-theta1.5-c5.csv")
gold$date = as.Date(gold$date)

GRID_SIZE = 400
kde.range = list(range(data$X),range(data$Y))
smoothed = bkde2D(data, 300, c(GRID_SIZE,GRID_SIZE), kde.range)


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

ksmooth = function(data, x) {
    smoothed = bkde2D(data, 300, c(GRID_SIZE,GRID_SIZE), kde.range)

    x$fhat = as.vector(t(smoothed$fhat)) #* nrow(data) # scale the density estimates by the total number of events in the last 7 days 
    return (x)
}

dates = unique(gold$date)
training.dates = dates[1:(length(dates)/2)]

x.khat = data.frame()

#date.range = seq(as.Date("2011-01-01"), as.Date("2011-06-01"), 1)
date.range = seq(as.Date("2011-01-01"), dates[length(dates)-27], 1)

for(day in 1:length(date.range)) {
    print(paste("day ",date.range[day]))
    month.start = date.range[day] #-28]
    month.end = date.range[day+27] #-1]
    ss = subset(data,(f$date >= month.start) & (f$date <= month.end))
    xhat = ksmooth(ss, x)

    xhat = subset(xhat, !is.na(xhat[,3]))
    x.bybeat = aggregate(xhat[,4], by=list(xhat[,3]), FUN=sum)
    names(x.bybeat) = c("beat","fhat")
    x.bybeat$date = date.range[day+28]
    x.khat = rbind(x.khat, x.bybeat)
}

merged = merge(x.khat, gold, by=c("beat","date"))
gold.chk = subset(gold, date %in% unique(x.khat$date))
merged.chk = merge(x.khat, gold.chk, by=c("beat","date"))

DEBUG = FALSE
if(DEBUG) {
    pdf("kde-movingtime-beatmap.pdf")
    for(day in c(29,79)) { #29:length(dates)) {
        beats.df.m = merge(beats.df, subset(merged,merged$date==dates[day]), by="beat", sort=FALSE, all.x=TRUE)
        row.names(beats.df.m) = row.names(beats.df)
        toplot = SpatialPolygonsDataFrame(as(beats, "SpatialPolygons"), data=beats.df.m)
        p1 = spplot(toplot, "on")
        p2 = spplot(toplot, "fhat")

        print(p1, position = c(0,.5,.5,1),more=T)
        print(p2, position = c(.5,.5,1,1))
    }
    dev.off()

    p1 = spplot(toplot, "on")
    p2 = spplot(toplot, "fhat")

    print(p1, position = c(0,.5,.5,1),more=T)
    print(p2, position = c(.5,.5,1,1))

}

training.data = subset(merged, merged$date %in% training.dates)
test.data = subset(merged, !(merged$date %in% training.dates))
fit = glm(on ~ fhat + factor(beat), data=training.data, family=binomial(link="logit")) 
fit.lmer = lmer(on ~ fhat + (1|as.factor(training.data$beat)), data=training.data, family=binomial)

predictions = predict.glm(fit, test.data, type="response")
predictions.lmer = predict(fit, test.data, type="response")

pred = prediction(predictions, test.data$on)
perf = performance(pred,"tpr","fpr")
auc = performance(pred,"auc")
print(round(auc@y.values[[1]],4))

pdf("kde-withtime-fixedeffects.pdf")
plot(perf, xlim=c(0,.15), ylim=c(0,.5), main=paste("KDE with a time component, AUC=", round(auc@y.values[[1]],4)))

lines(c(0,1),c(0,1),col=2)
dev.off()

if(FALSE){
predictions = predict.glm(fit,type="response") # ,predictions="???

pred = prediction(predictions, fit$y) #training.data$on)
perf = performance(pred,"tpr","fpr")
plot(perf, xlim=c(0,.15), ylim=c(0,.5), main="KDE with a time component")
lines(c(0,1),c(0,1),col=2)
}
