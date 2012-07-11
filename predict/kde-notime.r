# Method 1 from Amrut's paper: ignore time, use a kernel to smooth the violent crime data 
# in space, and then use density estimates to predict

f = read.csv("geocoded.csv")
f = subset(f,f$type=="VIOLENT")

library(sparr)
#library(spatstat)
library(spdep)
library(maptools)
library(sm)
#library(gpclib)


beats = readShapePoly("../fetchcalls/shp/cpd_beats.shp")
city = readShapePoly("../fetchcalls/shp/City_Boundary.shp")
city.bounds = as.owin(city)

data = f[,c("X","Y")]
pp = ppp(data$X,data$Y,window=city.bounds)
smoothed = density.ppp(pp)
plot(smoothed)

smooth = bivariate.density(pp, adaptive = FALSE)
smooth = bivariate.density(data, xrange=range(data$X), yrange=range(data$Y), pilotH=100, WIN=city.bounds, adaptive = FALSE)

pp = ppp(sample$X,sample$Y,window=city.bounds)
bw = LSCV.density(sample,WIN=city.bounds)
