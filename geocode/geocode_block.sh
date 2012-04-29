python geocode_block.py ../fetchcalls/311_sample.csv  > ../fetchcalls/311-geocoded_block.csv
python postprocess.py ../fetchcalls/311-geocoded_block.csv > ../input/311-calls-January2011-March2012-block.csv
