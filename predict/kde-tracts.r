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
#library(gpclib)
library("KernSmooth")
library(ROCR)


areals = readShapePoly("../fetchcalls/shp/Census_Tracts.shp")
areal.name = "CENSUS_TRA"
areals.centers = coordinates(areals)

#areals.df = as.data.frame(areals)
#areals.df = areals.df[,-3]
#names(areals.df)[3] = "areal"
#areals.df$areal = as.numeric(as.character(areals.df$areal))

data = f[,c("X","Y", "type")]
gold = read.csv("input/rats-tracts.csv") #gold-theta1.5-c5.csv")
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
smoothed.areals = over(sp, areals)
x = data.frame(xdatarange,ydatarange,smoothed.areals[,areal.name],as.vector(t(smoothed$fhat)))
# debug:
# library(fields)
# quilt.plot(x[,1],x[,2],x[,4],nx=GRID_SIZE,ny=GRID_SIZE)
names(x) = c("xdatarange","ydatarange","AREAL","fhat")
x$AREAL = as.numeric(as.character(x$AREAL))
streams = unique(f$type)

ksmooth = function(data, x, streams) {
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
#for(day in 1:(length(date.range)-27)) { 
x.byareals = foreach(day=1:(length(date.range) - 27)) %dopar% {
    print(paste("day ",date.range[day]))
    month.start = date.range[day] 
    month.end = date.range[day+27] 

    ss = subset(f,(f$date >= month.start) & (f$date <= month.end))
    xhat = ksmooth(ss[,c("X","Y","type")], x, streams) #c("Rodent"))

    xhat = subset(xhat, !is.na(xhat$AREAL))
    x.byareal = aggregate(xhat[,5:ncol(xhat)], by=list(xhat$AREAL), FUN=sum)
    names(x.byareal)[1] = "areal"
    x.byareal$date = date.range[day+28]
    if(MULTICORE) {
        return(x.byareal)
    } else {
        x.khat = rbind(x.khat, x.byareal)
    }
}

if(MULTICORE) {
    for(r in 1:length(x.byareals)) {
        x.khat = rbind(x.khat, x.byareals[[r]])
    }
}

#eval.fit(on ~ VIOLENT, "/tmp/test.pdf", "using violent w/ KDE", training.data, test.data)
#eval.fit(on ~ VIOLENT + factor(dow), "/tmp/test.pdf", "using violent w/ KDE", training.data2, test.data2)
#eval.fit(on ~ . -date -areal + factor(dow), "/tmp/test.pdf", "using violent w/ KDE", training.data2, test.data2)

eval.fit = function(eq, filename, title, training.data, test.data) {
    fit = glm(eq, data=training.data, family=binomial(link="logit"))
    predictions = predict.glm(fit, test.data, type="response")

    pred = prediction(predictions, test.data$on)
    perf = performance(pred,"tpr","fpr")
    auc = performance(pred,"auc")
    print(round(auc@y.values[[1]],4))

    pdf(filename)
    plot(perf, xlim=c(0,.15), ylim=c(0,.75), main=paste(title, "AUC=", round(auc@y.values[[1]],4)))

    lines(c(0,1),c(0,1),col=2)
    dev.off()
}


for(t in streams) {
    gold.name = paste("input/", t, "-tracts.csv", sep="")
    if(!file.exists(gold.name)) { 
        system(paste('python gold_standard.py --areal tract --stream "', t, '" --output ', gold.name, sep=""))
    }
    gold = read.csv(gold.name)
    gold$date = as.Date(gold$date)
    names(gold)[1] = "areal"
    merged = merge(x.khat, gold, by=c("areal","date"))

    #merged.scaled[,3:33] = apply(merged[,3:33], 2, function(x){ave(x,merged$areal, FUN=scale)})
#    merged.scaled[,3] = apply(merged[,c(1,3)], 2, function(x){ave(x,merged$areal, FUN=scale)})[,2]
    #merged.scaled[,3] = apply(as.data.frame(merged[,3]), 2, function(x){ave(x,merged$areal, FUN=scale)})
    q = ncol(merged) - 1
    merged[,3:q] = apply(merged[,3:q], 2, function(x){ave(x,merged$areal, FUN=scale)})

    merged$dow = as.factor(as.numeric((merged$date - as.Date("2011-01-29"))) %% 7)
    merged$areal = as.factor(merged$areal)

    training.data = subset(merged, merged$date %in% training.dates)
    test.data = subset(merged, !(merged$date %in% training.dates))

    eval.fit(on ~ . -areal -dow, paste("plots/", t,"-tracts.pdf",sep=""), paste("predicting",t), training.data, test.data)
}
