# Method 3 from Amrut's paper: use a kernel to smooth the violent crime data, plus other leading indicators, 
# from days -1 to days -28 to predict day 0

f = read.csv("geocoded.csv")
f$date = as.Date(f$date,format="%m/%d/%Y")
violent = subset(f,f$type=="VIOLENT")

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

data = f[,c("X","Y", "type")]
gold = read.csv("input/gold-theta1.5-c5.csv")
gold$date = as.Date(gold$date)

GRID_SIZE = 200
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
# library(fields)
# quilt.plot(x[,1],x[,2],x[,4],nx=GRID_SIZE,ny=GRID_SIZE)
names(x) = c("xdatarange","ydatarange","BEAT","fhat")
x$BEAT = as.numeric(as.character(x$BEAT))
streams = unique(f$type)

ksmooth = function(data, x) {
    for(t in streams) {
        smoothed = bkde2D(subset(data,data$type==t), 300, c(GRID_SIZE,GRID_SIZE), kde.range)
        x[,t] = as.vector(t(smoothed$fhat))
    }

#    x$fhat = as.vector(t(smoothed$fhat)) #* nrow(data) # scale the density estimates by the total number of events in the last 7 days 
    return (x)
}

dates = unique(gold$date)
training.dates = dates[1:(length(dates)/2)]


#date.range = seq(as.Date("2011-01-01"), as.Date("2011-03-01"), 1)
date.range = seq(as.Date("2011-01-01"), dates[length(dates)], 1)

x.khat = data.frame()
for(day in 1:(length(date.range)-27)) { 
    print(paste("day ",date.range[day]))
    month.start = date.range[day] 
    month.end = date.range[day+27] 

    ss = subset(f,(f$date >= month.start) & (f$date <= month.end))
    xhat = ksmooth(ss[,c("X","Y","type")], x)

    xhat = subset(xhat, !is.na(xhat$BEAT))
    x.bybeat = aggregate(xhat[,5:ncol(xhat)], by=list(xhat$BEAT), FUN=sum)
    names(x.bybeat)[1] = "beat"
    x.bybeat$date = date.range[day+28]
    x.khat = rbind(x.khat, x.bybeat)
}

merged = merge(x.khat, gold, by=c("beat","date"))
for(t in streams) {
    merged[,t] = scale(merged[,t])
}

merged.scaled = merged
merged.scaled[,3:33] = apply(merged[,3:33], 2, function(x){ave(x,merged$beat, FUN=scale)})

training.data = subset(merged, merged$date %in% training.dates)
test.data = subset(merged, !(merged$date %in% training.dates))

merged.scaled$dow = as.factor(as.numeric((merged.scaled$date - as.Date("2011-01-29"))) %% 7)
merged.scaled$beat = as.factor(merged.scaled$beat)

training.data2 = subset(merged.scaled, merged$date %in% training.dates)
test.data2 = subset(merged.scaled, !(merged$date %in% training.dates))

eval.fit(on ~ VIOLENT, "/tmp/test.pdf", "using violent w/ KDE", training.data, test.data)
eval.fit(on ~ VIOLENT + factor(dow), "/tmp/test.pdf", "using violent w/ KDE", training.data2, test.data2)
eval.fit(on ~ . -date -beat + factor(dow), "/tmp/test.pdf", "using violent w/ KDE", training.data2, test.data2)

eval.fit = function(eq, filename, title, training.data, test.data) {
    fit = glm(eq, data=training.data, family=binomial(link="logit"))
    predictions = predict.glm(fit, test.data, type="response")

    pred = prediction(predictions, test.data$on)
    perf = performance(pred,"tpr","fpr")
    auc = performance(pred,"auc")
    print(round(auc@y.values[[1]],4))

    pdf(filename)
    plot(perf, xlim=c(0,.15), ylim=c(0,.5), main=paste(title, "AUC=", round(auc@y.values[[1]],4)))

    lines(c(0,1),c(0,1),col=2)
    dev.off()
}


fit = glm(on ~ . -date -beat + factor(beat), data=training.data, family=binomial(link="logit")) 
fit = glm(on ~ . -date -beat, data=training.data, family=binomial(link="logit")) 
fit = glm(on ~ VIOLENT + VIOLENT * factor(beat), data=training.data, family=binomial(link="logit")) 
#fit = glm(on ~ . -date -beat, data=training.data, family=binomial(link="logit")) 

predictions = predict.glm(fit, test.data, type="response")

pred = prediction(predictions, test.data$on)
perf = performance(pred,"tpr","fpr")
auc = performance(pred,"auc")
print(round(auc@y.values[[1]],4))

pdf("kde-all-LIs-scaled.pdf")
plot(perf, xlim=c(0,.15), ylim=c(0,.5), main=paste("KDE with a time component, AUC=", round(auc@y.values[[1]],4)))

lines(c(0,1),c(0,1),col=2)
dev.off()

if(FALSE){
    fit = glm(merged[,c(6,3,4,5)], family=binomial(link="logit")) 
    fit = glm(on ~ . -date + factor(beat), data=merged, family=binomial(link="logit")) 
    predictions = predict.glm(fit,type="response") 

    pred = prediction(predictions, fit$y) #training.data$on)
    perf = performance(pred,"tpr","fpr")
    auc = performance(pred,"auc")
    print(round(auc@y.values[[1]],4))

    plot(perf, xlim=c(0,.15), ylim=c(0,.5), main=paste("KDE with a time component, AUC=", round(auc@y.values[[1]],4)))
    lines(c(0,1),c(0,1),col=2)
}
