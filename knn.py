import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing

df = pd.read_csv('Telecustomers.csv')

'''
K Nearest Neighbour is a supervised algorithm. It can solve both classification and regression algorithm. 
That takes a bunch of label points and uses them how to label other points.
'''

##print(df.head())
##print(df.columns)
##print(df.shape)


##print(df['custcat'].value_counts())


X = df.drop(['custcat'], axis = 1)
##print(X.head())

y = df['custcat']
##print(y.head())
##print(y.shape)

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
k = 4
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
Pred_y = neigh.predict(X_test)

print("Accuracy of model at K=4 is",metrics.accuracy_score(y_test, Pred_y))

#print(Pred_y)

#Now it’s time to improve the model and find out the optimal k value

Ks = 40
error_rate = []
# Will take some time
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
#plt.show()
print("Minimum error:-", min(error_rate), "at K =", error_rate.index(min(error_rate)))


acc = []
# Will take some time
from sklearn import metrics
for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(X_train,y_train)
    yhat = neigh.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, yhat))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue',linestyle='dashed',
         marker='o',markerfacecolor='red', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
#plt.show()
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))


### KNN is used broadly in the area of pattern recognition and analytical evaluation.



##### Evaluation

from sklearn.metrics import mean_squared_error, r2_score
print(y_test, Pred_y)
print(r2_score(y_test, Pred_y))
print(mean_squared_error(y_test,Pred_y))