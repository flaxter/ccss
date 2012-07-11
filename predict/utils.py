import sqlite3 as sql
from pandas import *

def load(filename):
    """ Loads and returns an sqlite3 database """

    return sql.connect(filename)

def select(db, areas="all", tracts="all", streams="all", start_date=None, end_date=None, fields="*"):
    """ To select a date range, start_date and end_date must both be non-None, and they must be strings formatted YYYY-MM-DD """
    cur = db.cursor()
    #db.row_factory = sql.Row

    where = "1"
    if areas != "all":
        areas = [str(a) for a in areas]
        where = " AND ".join([where,'area in (%s)'%(','.join(areas))])
    if tracts != "all":
        tracts = [str(t) for t in tracts]
        where = " AND ".join([where,'tract in (%s)'%(','.join(tracts))])
    if streams != "all":
        streams = ["'%s'"%s for s in streams]
        where = " AND ".join([where,'type in (%s)'%(','.join(streams))])
    if start_date != None and end_date != None: # currently you need them both
        where = "%s AND date between '%s' and '%s'"%(where,start_date,end_date)

    #print "SELECT %s from data where %s"%(fields,where)
    query = 'SELECT %s FROM data WHERE %s'%(fields, where)
    print query
    result = cur.execute(query)

    rows = result.fetchall()
    if len(rows) == 0:
        rows = None

    if fields == "COUNT()":
        return rows[0][0]
    return DataFrame(rows, columns="date type x y block beat tract area".split())

def query(db,q,cols=None):
    cur = db.cursor()
    result = cur.execute(q)
    rows = result.fetchall()
    if len(rows) == 0:
        return None
    elif len(rows[0]) == 1:
        return list(zip(*rows)[0])
    else: 
        return rows


if __name__ == "__main__":
    """ Extracts data from the database for use in other scripts, e.g. by R code
    TODO: add options, etc. """
    import sys
    db = load(sys.argv[1])

    data = select(db, streams=["sanitation"])
    f = open(sys.argv[2], "w")
    data.to_csv(f, index=False)
    f.close()
