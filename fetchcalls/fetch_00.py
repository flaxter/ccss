import os

data = {
    "vacant": "7nii-7srd",
    "sanitation": "me59-5fac",
    "tree-debris": "mab8-y9h3", 
    "tree-trims":"uxic-zsuj",	
    "potholes": "7as2-ds3y",
    "rodent": "97t6-zrhs",
    "grafitti": "hec5-y4x5",
    "lights-all": "zuxi-7xem",
    "lights-one": "3aav-uy2v",
    "abandoned-vehicle": "3c9v-pnva",
    "alley-lights": "t28b-ys7j",
    "garbagecarts": "9ksk-na4q",
#    "all-crime": "ijzp-q8t2"
}

if __name__ == "__main__":
    for name in data.keys():
        c = "wget http://data.cityofchicago.org/api/views/%s/rows.csv?accessType=DOWNLOAD -O data/%s.csv -o logs/%s-log.csv --background"%(data[name],name,name)
        print c
        os.system(c)

    print "run 01-cleanup.py when all the wgets finish"
    print "to check:"
    print "ps ax | grep wget"
