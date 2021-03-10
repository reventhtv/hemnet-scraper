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


headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36',
            'Cache-Control': 'no-cache'
        }

driver = webdriver.Chrome(r"C:\Users\rethi1\Downloads\chromedriver.exe") # This will open the Chrome window
sleep(2)


url = "https://www.hemnet.se/salda/bostader?location_ids%5B%5D=474035"

driver.get(url)
sleep(3)

page = requests.get(url, headers=headers)
soup = soup(page.content, 'html.parser')

response = requests.get(url)

element = soup.findAll("div", {"class": "sold-property-listing__location"})
#print(element[0])

'''
span = soup.findAll("span", {"class": "item-link"})
print(span)

span1 = span[1].getText()
print(span1)
'''
#print(element)
#print(element[0])

location = element[1].findAll("span", {"class": "item-link"})[1].getText()
#print(location)


listing_size = soup.findAll("div", {"class": "sold-property-listing__size"})
#print(listing_size[0].getText().splitlines())

listing_price = soup.findAll("div", {"class": "sold-property-listing__price"})
#print(listing_price[0].getText().splitlines())

price_change = soup.findAll("div", {"class": "sold-property-listing__price-change"})
#print(price_change[0].getText().splitlines())

listing_broker = soup.findAll("div", {"class": "sold-property-listing__broker"})
#print(listing_broker[0].getText().splitlines())

expand_hits = soup.findAll("a", {"class": "sold-property-listing"})
#print(expand_hits[0].find("div", {"class": "sold-property-listing__location"}).findAll("span", {"class": "item-link"})[1].getText())
#print(expand_hits[0].find('div', class_='sold-property-listing__size').getText().splitlines())

#print(expand_hits[1])
#print(expand_hits[0].find("div", {"class": "sold-property-listing__location"}).findAll("span", {"class": "item-link"})[1].getText())


apartments = []
for hit_property in expand_hits:
    #element = soup.findAll("div", {"class": "sold-property-listing__location"})
    place_name = expand_hits[0].find("div", {"class": "sold-property-listing__location"}).findAll("span", {"class": "item-link"})[1]
    size = hit_property.find("div", {"class": "sold-property-listing__size"})
    monthly = hit_property.find("div", {"class": "sold-property-listing__price"})
    final_str = place_name.getText()+", "+monthly.getText()+", "+size.getText()
    #print(final_str)
    apartments.append(final_str)




print(apartments)
unclean = apartments[1].splitlines()

unclean = [e.strip() for e in unclean]
clean = [e for e in unclean if len(e) != 0]

#print(unclean, len(unclean))
print(clean, len(clean))
print(clean[0])
print(clean[1])
print(clean[2])
print(clean[3])
print(clean[4])
print(clean[5])
print(clean[6])
print(clean[7])



import re
clean_apartments = []
missed_clean_up = []
##clean_apartments.append({'price_per_size': int(re.sub('[^0-9]','',clean[3]))})
##print(clean_apartments)
for aprt in apartments:
    unclean = aprt.splitlines()
    print(unclean)
    unclean = [e.strip() for e in unclean]
    print(unclean)
    clean = [e for e in unclean if len(e) != 0]
    print(clean)
    if len(clean) == 9:
        #print("GOTCHA")
        try:
            clean_apartments.append(
                {'region': clean[0].replace(",",""),
                 'price': int(re.sub('[^0-9]','',clean[2])),
                 'price_per_size': int(re.sub('[^0-9]','',clean[4])),
                 'size': float(clean[6].replace(u'\xa0', ' ').replace(" m²", "").replace(",", ".")),
                 'rooms': float(clean[7].replace(u'\xa0', ' ').replace(" rum", "").replace(",", ".")),
                 'rent': int(re.sub('[^0-9]', '', clean[8]))
                })
            #print(clean_apartments)
        except ValueError:
            #print("Oops! Not a valid number. Try again!")
            missed_clean_up.append(clean)
    else:
        missed_clean_up.append(clean)
        #print("GOTCHA 2")

#print(clean_apartments)
#print(missed_clean_up)

df = pd.DataFrame(data=clean_apartments)
#print(df.head())
df_size = df.size
#print(df_size)

import nltk
nltk.download()

df_byPrice = df.sort_values(by='price', ascending=False)
#print(df_byPrice.head())


import numpy as np
import tensorflow as tf
from tensorflow import keras

df.plot.scatter(x='size',y='price',c='price_per_size', colormap='viridis')


df.plot.scatter(x='size',y='price')

df.to_csv('test_data.csv',encoding='utf-8',index=False)

from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df)

#print(enc.categories_)
edsviken_df = df[(df['region'] == "Kalkbrottet")]
#print(edsviken_df)

edsviken_df.plot.scatter(x='size',y='price')
plt.show()

data = pd.read_csv("test_data.csv",header=0,index_col=False)


#print(data.head())
#print(data.columns.tolist())

data.plot.scatter(x='size',y='price')
plt.show()
