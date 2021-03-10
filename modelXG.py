import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os,re
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
from xgboost import XGBRegressor

from scipy import stats
from scipy.stats import skew,kurtosis

from sklearn.preprocessing import StandardScaler,scale
from sklearn.model_selection import train_test_split,RandomizedSearchCV,KFold
from sklearn.metrics import mean_squared_error, r2_score,mean_squared_log_error
from sklearn.linear_model import LinearRegression
from feature_engine.encoding import OrdinalEncoder
from sklearn.model_selection import train_test_split # data split
from termcolor import colored as cl # text customization

""" Analyse Outliers """
def bivrt_df_scatter(x,y,data=None,title=None,xlabel = None,ylabel=None):
    if isinstance(data,pd.DataFrame):
      df = pd.concat([data[x], data[y]], axis=1)
      df.plot.scatter(x=x, y=y,title = title )
    else:
      plt.scatter(x=x, y=y)
      plt.title(title)
      plt.xlabel(xlabel)
      plt.ylabel(ylabel)
    plt.show()


df = pd.read_csv("malmo_house_price_clean.csv")
df.drop_duplicates()
print(df.columns)
print(df.shape)
print(df.head())
#df_train = df[0:1600]

df['rent'].replace(' ', np.nan, inplace=True)
df.dropna(subset=['rent'], inplace=True)
df['rent'] = df['rent'].astype(float)
df.rename(columns={'price per sq mtr':'pricepersqmtr'}, inplace=True)
print(df.head())
df = df[~df.pricepersqmtr.str.contains("specified")]
df['pricepersqmtr'] = df['pricepersqmtr'].astype(float)
print(cl(df.dtypes, attrs = ['bold']))

#Identify the relationship across all the variables
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, annot = True, cmap = 'magma')
plt.show() #not mix with other similar graph


#Price is the output for later prediction model
#It is important that need to check its properties. Ex: min > 0 , mean no abnormal price happen
#df_train.price.describe()
print(df.rent.describe())
#Plot histogram (Normal distribution)
sns.distplot(df.price).set_title('Analysis Significant Output')
plt.show()

""" Analysis The Relationship Btw Output(SalePrice) and Correlated Variables  (Train data only)   """
#Get top n largest value order by SalePrice,then get the dataframe index because its  position changed.
cols = corrmat.nlargest(10, 'price').index
corrmat2 = df[cols].corr() #pandas method; .corr() : direct use for dataframe
sns.set(font_scale=1)
sns.heatmap(corrmat2, annot=True,annot_kws={'size': 9.5}, square=True, fmt='.2f')
plt.show()

#Pair plots between 'SalePrice' and correlated variables
sns.set() #RESET
sns.pairplot(df[cols], size = 1.5)
plt.show()

""" Analyse Outliers """
#----(Train data only)----

#----Bivariate analysis----
#By refering pairplot above, 'rent' and 'size' will be check with 'price'
#bivariate analysis price - rent
bivrt_df_scatter('rent','price',df,title = 'Analyse Outlier')
#bivariate analysis price - size
bivrt_df_scatter('size','price',df,title = 'Analyse Outlier')
plt.show()



""" Check test dataset """


#df.rename(columns={'price per sq mtr':'pricepersqmtr'}, inplace=True)
#df = df[~df.pricepersqmtr.str.contains("specified")]
#df['pricepersqmtr'] = df['pricepersqmtr'].astype(float)
#df['price'] = df['price'].astype(float)
#df['rent'] = df['rent'].astype(float)
print(cl(df.dtypes, attrs = ['bold']))

#Regularize data set
df.pricepersqmtr = df.pricepersqmtr / 10000
df.price = df.price / 1000000
df.rent = df.rent / 1000

print(df.head())

X = df.drop(columns=['price'], axis = 1)
Y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


#Encoding the regions
regions_df = np.asarray(X['region']).reshape(1,-1)
enc = OrdinalEncoder(encoding_method='ordered', variables=['region'])
enc.fit(X_train,y_train)

X_train_enc = enc.transform(X_train)
X_test_enc = enc.transform(X_test)

#fit model no training data
regressor = XGBRegressor(
    n_estimators=100,
    reg_lambda=1,
    gamma=0,
    max_depth=3
)
regressor.fit(X_train_enc,y_train)

#make predictions for test data
y_pred = regressor.predict(X_test_enc)
predictions = [round(value) for value in y_pred]

#Re-Normalizing price by multiplying with 1000000
price_predictions = y_pred * 1000000
print(len(y_pred))
print(y_pred)
print(price_predictions)
# evaluate predictions
mse = mean_squared_error(y_test, predictions)
print("Mean Square Error: %.2f%%" % mse)
r2score = r2_score(y_test,predictions)
print("R-squared error: %.2f%%" % r2score)
msle = mean_squared_log_error(y_test,predictions)
print("mean-squared log error: %.2f%%" % msle)

# Save model and encoder
np.save('regions.npy', enc.encoder_dict_)

with open('regions.json', 'w', encoding='utf8') as fp:
    json.dump(enc.encoder_dict_, fp, ensure_ascii=False)

regressor.save_model('hemnet-pred.model')