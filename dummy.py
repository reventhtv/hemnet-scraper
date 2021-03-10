from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
import requests, time
from bs4 import BeautifulSoup as soup
from time import sleep, strftime
from random import randint
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import smtplib
from email.mime.multipart import MIMEMultipart
from urllib.request import urlopen as uReq
import requests
import csv
import json
import matplotlib.pyplot as plt


url = "https://www.hemnet.se/salda/bostader?location_ids%5B%5D=17989"
page = requests.get(url, headers = {'User-Agent':'Mozilla/5.0'})
soup = BeautifulSoup(page.content,'html.parser')

apartments = []
for result in soup.select('.sold-results__normal-hit'):
    nodes = result.select('.sold-property-listing__location h2 + div span')
    if len(nodes)==2:
        place = nodes[1].text.strip().replace(",","")
    else:
        place = 'not specified'

    #print(place)
    monthly = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__price').text).split("2021")[1]
    #print(monthly)
    sold = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__price').text).split("Såld")[0].split("Slutpris")[1].replace(" ", "")

# Slutpris 3 760 000 kr Såld 5 mars 2021
    info = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__size').text)
    #print(result.select_one('.sold-property-listing__size'))
    print(info)

    if len(result.select_one('.sold-property-listing__size')) >= 3:
        #area = re.findall(r'[A-Za-z]+|\d+', info)
        area =  info.split(" m²")[0].replace(" ", "")
        rooms = info.split(" m²")[1].split(" rum")[0]
        rent = info.split(" kr/mån")[0].split(" rum")[1].replace(" ", "")
    else:
        area = 'not specified'
        rooms = 'not specified'
        rent = 'not specified'


    #text = str(info.replace(u'\xa0', ' ').replace(" m²", "").replace(",", "."))
    #print(text)


    #rent = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__price').text)
    final_str = place + info + rent
    #print(final_str)
    apartments.append({'region': place,
                       'area': area,
                       'rooms': rooms,
                       'rent': rent,
                       'Sold': int(re.sub('[^0-9]','',sold)),
                       'price per sq meter': monthly.replace(" kr/m²","").replace(" ", "")
                 })


print(apartments)

df = pd.DataFrame(data=apartments)
print(df.head())
df_size = df.size
print(df_size)
#df.replace('"', '', regex=True)
df['area'] = df['area'].str.replace(r"[\"\',]", '')
df['rooms'] = df['rooms'].str.replace(r"[\"\',]", '')
df = df[~df.rent.str.contains("tomt")]
df.to_csv('malmo_house_prices.csv',encoding='utf-8',index=False)
print(df.info())
import re

df.plot.scatter(x='area',y='Sold', colormap='viridis')
plt.show()

#df['result'] = df['result'].str.replace(r'\D', '')

'''
print(soup.select('.sold-results__normal-hit'))
for result in soup.select('.sold-results__normal-hit'):
    place_name = re.sub(r'\s{2,}',' ', result.select_one('.sold-property-listing__location h2 +div span').text)
    ##print(soup.select('.sold-property-listing__size'))
    size = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__size').text)
    rent = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__price').text)
    final_str = place_name+size+rent
    print(final_str)
    
'''

