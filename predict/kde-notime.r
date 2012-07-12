# Method 1 from Amrut's paper: ignore time, use a kernel to smooth the violent crime data 
# in space, and then use density estimates to predict

f = read.csv("geocoded.csv")
f = subset(f,f$type=="VIOLENT")

library("doMC")
registerDoMC()

library(sparr)
#library(spatstat)
library(spdep)
library(maptools)
library(sm)
#library(gpclib)
library("KernSmooth")


beats = readShapePoly("../fetchcalls/shp/cpd_beats.shp")
city = readShapePoly("../fetchcalls/shp/City_Boundary.shp")
beats.centers = coordinates(beats)
city.bounds = as.owin(city)

data = f[,c("X","Y")]

smoothed = bkde2D(data, 300, c(1000,1000))
xdatarange <- sort(rep(smoothed$x1,1000))
ydatarange <- rep(smoothed$x2,1000)

sp = SpatialPoints(data.frame(xdatarange,ydatarange))
smoothed.beats = over(sp, beats)

x = data.frame(xdatarange,ydatarange,smoothed.beats$BEAT_NUM,as.vector(t(smoothed$fhat)))
# to check:
# library(fields)
# quilt.plot(x[,1],x[,2],x[,4])

x = subset(x, !is.na(x[,3]))
x.bybeat = aggregate(x[,4], by=list(x[,3]), FUN=sum)
x.bybeat[,1] = as.numeric(as.character(x.bybeat[,1]))
gold = read.csv("input/gold-theta1.5-c5.csv2")
gold.average = aggregate(gold$on,by=list(gold$beat),FUN=mean)
names(gold.average) = c("beat","xbar")
names(x.bybeat) = c("beat","fhat")

all.data = merge(x.bybeat,gold, by="beat")
dates = unique(all.data$date)
training.dates = dates[1:(length(dates)/2)]

training.data = subset(all.data, all.data$date %in% training.dates)
test.data = subset(all.data, !(all.data$date %in% training.dates))
fit = glm(training.data$on ~ training.data$fhat, family=binomial(link="logit"))

predictions = predict.glm(fit, test.data, type="response")
library(ROCR)

pred = prediction(predictions, test.data$on)
perf = performance(pred,"tpr","fpr")
pdf("kde-notime.pdf")
plot(perf, xlim=c(0,.15), ylim=c(0,.5), main="KDE without a time component")
lines(c(0,1),c(0,1),col=2)
dev.off()

if(FALSE){ 
validation = merge(x.bybeat,gold.average, by="beat")

d = unique(gold$date)[1]
validation.day = merge(x.bybeat,subset(gold,gold$date==d), by="beat")
fit = glm(validation.day[,4] ~ validation.day[,2], family=binomial(link="logit"))

#pp = ppp(data$X,data$Y,window=city.bounds)
sample <- data[sample(1:nrow(data), 1000, replace=FALSE),]
pp = ppp(sample$X,sample$Y,window=city.bounds)
h = 30

#smoothed = density.ppp(pp.sampled)
#plot(smoothed)

q = foreach(h=c(1,5,10,15,20,25,30,50)) %dopar% {
detach("package:sparr",unload=TRUE)
library(sparr)

    smooth = bivariate.density(pp, pilotH=h, adaptive = TRUE, intensity=TRUE, res=2)
    pdf(paste("plots/pilotH-",h,".pdf",sep=""))
    plot(smooth)
    dev.off()
}

smooth = bivariate.density(pp, adaptive = FALSE)
smooth = bivariate.density(data, xrange=range(data$X), yrange=range(data$Y), pilotH=100, WIN=city.bounds, adaptive = FALSE)

smooth = bivariate.density(pp, pilotH=100, adaptive = TRUE, intensity=TRUE,atExtraCoords=beats.centers)

bw = LSCV.density(sample,WIN=city.bounds)
}
