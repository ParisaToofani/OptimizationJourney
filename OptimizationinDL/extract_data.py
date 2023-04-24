from bs4 import BeautifulSoup, SoupStrainer
import requests
from urllib.parse import urljoin
import re
import os
import urllib

url = "https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2010/"

page = requests.get(url)    
data = page.text
soup = BeautifulSoup(data)
i = 0
for link in soup.find_all('a'):
    # if i >= 4:
        # filename = os.path.join(folder_location,link['href'].split('/')[-1])
        # with open('C:\\AllMyCodes\\OptimizationProject\\OptimizationJourney\\data', 'wb') as f:
        #     f.write(requests.get(urljoin(url + link.get('href'))).content)
    try:
        # print(url+link.get('href'))
        # url + link.get('href')
        # with open(link.get('href'), 'wb') as f:
        #     f.write(requests.get('https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2010/HURSAT_b1_v06_2010002S09096_EDZANI_c20170721.tar.gz').content)
        # files.download(url + link.get('href'))
        print(url + link.get('href'))
        with open(link.get('href'), 'wb') as f:
            print('hi')
            f.write(requests.get('https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2010/HURSAT_b1_v06_2010002S09096_EDZANI_c20170721.tar.gz').content)
            # f.write(requests.get(url + link.get('href')).content)
    except:
        print('Hi')

    i += 1
# https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2010/HURSAT_b1_v06_2010002S09096_EDZANI_c20170721.tar.gz  
# https://www.ncei.noaa.gov/data/hurricane-satellite-hursat-b1/archive/v06/2010/HURSAT_b1_v06_2010002S09096_EDZANI_c20170721.tar.gz