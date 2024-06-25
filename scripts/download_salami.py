# Modified from a script by Oriol Nieto provided in the SALAMI dataset

import os
import argparse
import requests
import csv

def download(url, localName):
    """Downloads the file from the url and saves it as localName."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(localName, 'wb') as f:
            f.write(response.content)
    else:
        print("Could not download:", url)

def process( csv_file, output_dir ):
    """Main process function to download all mp3s from the csv_file
    and put thm in the ouput_dir."""
    f = open( csv_file, "r" )
    file_reader = csv.reader(f)
    for fields in file_reader:
        id = fields[0]
        url = fields[4]
        print("Downloading: ", id, url)
        try:
            download( url, os.path.join(output_dir, id + ".mp3") )
        except Exception as e:
            print("Could not retrieve:", id, url)
            print("Exception:", e)
    f.close()

def main():
    process("resources/salami-data-public/metadata/id_index_internetarchive.csv", "resources/salami-data-public/music")

if __name__ == "__main__":
    main()
