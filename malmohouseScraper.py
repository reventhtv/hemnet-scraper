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
import seaborn as sb
import matplotlib.pyplot as plt

for i in range(0,500):
    url = "https://www.hemnet.se/salda/bostader?location_ids%5B%5D=17989&page={}".format(i)
    page = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(page.content, 'html.parser')

    apartments = []
    for result in soup.select('.sold-results__normal-hit'):
        nodes = result.select('.sold-property-listing__location h2 + div span')
        if len(nodes) == 2:
            place = nodes[1].text.strip().replace(",", "")
        else:
            place = 'not specified'

        print(place)
        #print(result)
        house_price = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__price').text).split("Såld")[0].split("Slutpris")[1].replace(" ","").replace("kr","")
        price = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__price').text).split("Såld")[1]
        print(house_price)
        price = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__price').text).split("Såld")[1]
        #print(price)

        #per_sq_meter = price.split("2021")

        if ("kr/m²") in price:
            if ("2021") in price:
                price_per_sq_meter = price.split("2021")[1].replace(" ","").replace(" kr/m²","")
            elif ("2020") in price:
                price_per_sq_meter = price.split("2020")[1].replace(" ","").replace(" kr/m²","")
            else:
                price_per_sq_meter = "No 2020 & 2021"
        else:
            price_per_sq_meter = "price not specified"

        print(price_per_sq_meter)

        info = re.sub(r'\s{2,}', ' ', result.select_one('.sold-property-listing__size').text)
        # print(info)

        if ("rum") in info:
            room = info.split(" m²")
            if len(room) >= 2:
                rooms = info.split()[2].replace(",", ".")
            else:
                rooms = "rooms not specified"
        else:
            room = "rooms not specified"

        size = info.split()[0].replace(",", ".")
        rent = info.split("rum")
        if len(rent) >= 2:
            monthly = rent[1].replace(" kr/mån", "").replace(" ", "")
        else:
            monthly = "Rent not specified"

        print(float(size))
        print(rooms)
        print(monthly)

        apartments.append({
            "region": place,
            "price": house_price,
            "price per sq meter": price_per_sq_meter,
            "rooms": float(rooms),
            "size": size,
            "rent": monthly
        })

    df = pd.DataFrame(data=apartments)
    # print(df.head())
    df_size = df.size
    # print(df_size)
    df.to_csv('malmo_house_prices.csv', encoding='utf-8', index=False, mode='a', header=False)


df_read = pd.read_csv("malmo_house_prices.csv", names = ['region', 'price', 'price per sq mtr', 'rooms', 'size','rent'])
print(df_read.head())
df_read = df_read[~df_read.rent.str.contains("tomt")]
df_read = df_read[~df_read.rent.str.contains("biarea")]
df_read = df_read[~df_read.rent.str.contains("specified")]
df_read = df_read[~df_read.region.str.contains("specified")]
df_read.drop_duplicates()
df_read['price per sq mtr'].replace(' ', np.nan, inplace=True)
df_read.dropna(subset=['price per sq mtr'], inplace=True)
df_read.to_csv('malmo_house_price_clean.csv', encoding='utf-8', index=False)