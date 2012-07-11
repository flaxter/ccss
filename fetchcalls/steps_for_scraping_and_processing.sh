#!/bin/sh

# scrape all 311 data, concatenate it
python fetch.py 311.csv

# geocode 311.csv

python --no-save < geocode.r 
#python geocode.py 311.csv > 311-geocoded.csv
#python postprocess.py 311-geocoded.csv > 311-census.csv

R --no-save < long_to_wide.R

# put it into a sqlite database

rm 311.db
sqlite3 311.db < process.sqlite3

# clean, aggregate data for summary statistics, conventional analysis 


