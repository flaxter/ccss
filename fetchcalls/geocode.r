library(maptools)
library(spdep)

tracts = readShapePoly("shp/CensusTractsTIGER2010.shp")
#tracts2000 = readShapePoly("../../../PHDCN/shp/Census_Tracts.shp")
beats = readShapePoly("shp/cpd_beats.shp")
areas = readShapePoly("shp/CommAreas.shp")
#blocks = readShapePoly("shp/Census Blocks.shp")

#plot(areas, border="green", axes=TRUE, las=1)
#par(new=TRUE)
#plot(tracts, border="blue", axes=TRUE, las=1)
#par(new=TRUE)
#plot(beats, border="red", axes=FALSE, las=1)
#par(new=TRUE)
#plot(blocks, border="red", axes=FALSE, las=1)

f = read.csv("311.csv")

data = f[complete.cases(f),]

ss = data$type
counts = aggregate(ss, list(ss), FUN=length)
include = counts[,1][counts[,2] > 200] 
data = subset(data, data$type %in% include)
data = subset(data, data$type != "")
data$type[data$type == "INTERFERE WITH PUBLIC OFFICER"] = "INTERFERENCE WITH PUBLIC OFFICER"
violent = c("HOMICIDE", "CRIM SEXUAL ASSAULT", "ROBBERY", "ASSAULT", "BATTERY")
data$type = factor(data$type, levels = c(levels(data$type), "VIOLENT", "Lights", "Rodent", "Garbage"))

data$type[data$type %in% violent] = "VIOLENT"

data$type[data$type == "Street Lights - All/Out"] = "Lights"
data$type[data$type == "Rodent Baiting/Rat Complaint"] = "Rodent"
data$type[data$type == "Garbage Cart Black Maintenance/Replacement"] = "Garbage"

latlong = data[,c(3,4)]

sp = SpatialPoints(latlong)
geocoded.tracts = over(sp,tracts)
geocoded.tracts2000 = over(sp,tracts2000)
geocoded.beats = over(sp,beats)
#geocoded.blocks = over(sp,blocks)

#geocoded = data.frame(data,geocoded.blocks$CENSUS_BLO,geocoded.beats$BEAT_NUM,geocoded.tracts$TRACTCE10,geocoded.tracts$COMMAREA)
geocoded = data.frame(data,geocoded.beats$BEAT_NUM,geocoded.tracts$TRACTCE10,geocoded.tracts$COMMAREA)
#names(geocoded) = c(names(data),"block","beat","tract","area")
names(geocoded) = c(names(data),"beat","tract","area")
write.csv(geocoded, "geocoded-unique.csv",row.names=FALSE, quote=FALSE)

geocoded = data.frame(data, as.integer(substr(as.character(geocoded.tracts2000$CENSUS_TRA), 1,4)))
names(geocoded) = c(names(data),"tract")
write.csv(geocoded, "geocoded-census2000.csv",row.names=FALSE, quote=FALSE)

## convert to graph for INLA

#nb = poly2nb(tracts)
#nb2INLA(file="chicago_tracts.graph",nb)

## aggregation

data.violent = subset(data,data$type=="VIOLENT")[,c("date","beat")]
write.csv(data.violent,"violent-by-beat.csv",row.names=FALSE,quote=FALSE)

#counts = aggregate(data.violent, list(data.violent$date,data.violent$beat), FUN=length)
