#!/bin/sh

# run this script to fetch, clean, and process 311 calls from 
# data.cityofchicago.org

python fetch.py
python cleanup_311_data.py 311.csv > 311-latlong.csv
python cleanup_311_data.py 311.csv > 311-census.csv
python filter_data.py > 311-census-violent.csv
