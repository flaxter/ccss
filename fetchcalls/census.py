from lxml import etree
import shapely
import re
try:
    tree = etree.parse(open("../censustracts2010.kml","r"))
except:
    tree = etree.parse(open("censustracts2010.kml","r"))

from shapely.geometry import Point, Polygon
def topoint(s):
    return (float(s[0]),float(s[1]))


try:
    tree = etree.parse(open("../censustracts2010.kml","r"))
except:
    tree = etree.parse(open("censustracts2010.kml","r"))
id = ''
tracts = {}
tracts_to_area = {}

area_re = r'Chicago Community Area</td>\r\r\n<td>([0-9]+)'
tract_re = r'Census Tract ([0-9.]+)'
#print "id,area,tract"



for action, e in context:
    if "Placemark" in e.tag:
        id = e.values()[0][3:]
        coords = e.iterdescendants(tag='{http://www.opengis.net/kml/2.2}coordinates').next().text
	html = e.iterdescendants(tag='{http://www.opengis.net/kml/2.2}description').next().text
	match = re.search(area_re, html)
	area = match.group(1)
	match = re.search(tract_re, html)
	tract = match.group(1)
	#print "%s,%s,%s"%(id, area, tract)
	
        tracts[id] = Polygon([topoint(c.split(',')) for c in coords.split()[1:]])
	tracts_to_area[id] = [tract,area]

