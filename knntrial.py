import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df = pd.read_csv('Telecustomers.csv')

from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor

X = df.drop(['custcat'], axis = 1)
##print(X.head())

y = df['custcat']
##print(y.head())
##print(y.shape)



knn = KNeighborsClassifier(n_neighbors=4)
y_pred = cross_val_predict(knn, X, y, cv=5)
print(y_pred.shape)

from sklearn.metrics import mean_squared_error, r2_score

print(r2_score(y,y_pred))
print(mean_squared_error(y,y_pred))


#Now let's split the data to train data and test data
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)

print("Accuracy of model at K=4 is",metrics.accuracy_score(y_test, Pred_y))

#Now it’s time to improve the model and find out the optimal k value

Ks = 40
error_rate = []
# Will take some time
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    y_pred = cross_val_predict(knn, X, y, cv=5)
    error_rate.append(mean_squared_error(y,y_pred))

import matplotlib.pyplot as plt

plt.plot(range(1,40), error_rate)
plt.show()
print("minimum mean squared error", min(error_rate), "is at k=", error_rate.index(min(error_rate)))

