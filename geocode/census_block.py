from lxml import etree
import shapely
import re
#tree = etree.parse(open("../censusblocks2010.kml","r"))

from shapely.geometry import Point, Polygon
def topoint(s):
    return (float(s[0]),float(s[1]))


context = etree.iterparse(open("../censusblocks2010.kml","r"))
id = ''
blocks = {}
#tracts_to_area = {}

block_re = r'TRACT_BLOCK</td>\r\r\n<td>([0-9]+)'

#print "id,area,tract"


count = 0
for action, e in context:
    if "Placemark" in e.tag:
        id = e.values()[0][3:]
        coords = e.iterdescendants(tag='{http://www.opengis.net/kml/2.2}coordinates').next().text
	html = e.iterdescendants(tag='{http://www.opengis.net/kml/2.2}description').next().text
	#match = re.search(area_re, html)
	#area = match.group(1)
	match = re.search(block_re, html)
	block = match.group(1)
	blocks[block] = Polygon([topoint(c.split(',')) for c in coords.split()[1:]])
	count +=1 
	#if(count  < 2):
		#print block,blocks[block]
	
	
        
	

