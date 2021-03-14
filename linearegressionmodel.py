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
from termcolor import colored as cl # text customization
import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sb # visualization
from termcolor import colored as cl # text customization

from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm

from sklearn.metrics import explained_variance_score as evs # evaluation metric
from sklearn.metrics import r2_score as r2 # evaluation metric

df = pd.read_csv("skane_house_price_clean.csv")
df['rent'].replace(' ', np.nan, inplace=True)
df.dropna(subset=['rent'], inplace=True)
df['rent'] = df['rent'].astype(float)
df['price per sq mtr'].replace(' ', np.nan, inplace=True)
df.dropna(subset=['price per sq mtr'], inplace=True)

X_var = df[['rooms','size','rent']].values
y_var = df['price'].values

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)

#print(cl('X_train samples : ', attrs = ['bold']), X_train[0:5])
#print(cl('X_test samples : ', attrs = ['bold']), X_test[0:5])
#print(cl('y_train samples : ', attrs = ['bold']), y_train[0:5])
#print(cl('y_test samples : ', attrs = ['bold']), y_test[0:5])

# MODELING

# 1. OLS

ols = LinearRegression()
ols.fit(X_train, y_train)
ols_yhat = ols.predict(X_test)

#print(X_test)
#print(ols_yhat)


df_ols = pd.DataFrame(data=ols_yhat)
df_ols.columns = ['ols predicted']

print(df_ols)
#print(cl(df_ols.dtypes, attrs = ['bold']))

print(cl('Explained Variance Score of OLS model is {}'.format(evs(y_test, ols_yhat)), attrs = ['bold']))
print(cl('R-Squared of OLS model is {}'.format(r2(y_test, ols_yhat)), attrs = ['bold']))

# 2. Ridge

ridge = Ridge(alpha = 0.5)
ridge.fit(X_train, y_train)
ridge_yhat = ridge.predict(X_test)

#print(ridge_yhat)

df_ridge = pd.DataFrame(data=ridge_yhat)
df_ridge.columns = ['ridge predicted']
print(df_ridge)

print(cl('Explained Variance Score of Ridge model is {}'.format(evs(y_test, ridge_yhat)), attrs = ['bold']))
print(cl('R-Squared of Ridge model is {}'.format(r2(y_test, ridge_yhat)), attrs = ['bold']))

# 3. Lasso

lasso = Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)
lasso_yhat = lasso.predict(X_test)

df_lasso = pd.DataFrame(data=lasso_yhat)
df_lasso.columns = ['lasso predicted']
print(df_lasso)

print(cl('Explained Variance Score of Lasso model is {}'.format(evs(y_test, lasso_yhat)), attrs = ['bold']))
print(cl('R-Squared of Lasso model is {}'.format(r2(y_test, lasso_yhat)), attrs = ['bold']))

# 4. Bayesian

bayesian = BayesianRidge()
bayesian.fit(X_train, y_train)
bayesian_yhat = bayesian.predict(X_test)

df_bayesian = pd.DataFrame(data=bayesian_yhat)
df_bayesian.columns = ['bayesian predicted']
print(df_bayesian)

print(cl('Explained Variance Score of Bayesian model is {}'.format(evs(y_test, bayesian_yhat)), attrs = ['bold']))
print(cl('R-Squared of Bayesian model is {}'.format(r2(y_test, bayesian_yhat)), attrs = ['bold']))


# 5. ElasticNet

en = ElasticNet(alpha = 0.01)
en.fit(X_train, y_train)
en_yhat = en.predict(X_test)

df_en = pd.DataFrame(data=en_yhat)
df_en.columns = ['en predicted']
print(df_en)

print(cl('Explained Variance Score of ElasticNet is {}'.format(evs(y_test, en_yhat)), attrs = ['bold']))
print(cl('R-Squared of ElasticNet is {}'.format(r2(y_test, en_yhat)), attrs = ['bold']))
