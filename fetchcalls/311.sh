#!/bin/sh

# scrape all 311 data, concatenate it
python fetch.py 311.csv

# geocode 311.csv
python geocode_311_data.py 311.csv > 311-geocoded.csv
python postprocess.py 311-geocoded.csv > 311-census-violent.csv

