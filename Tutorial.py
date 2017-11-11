import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

'''
gl = pd.read_csv('game_logs.csv')
gl.info(memory_usage = 'deep')
print(gl.loc[0])
'''

'''
s = pd.Series([1,3,5,np.nan,6,8])
dates = pd.date_range('20130101', periods=6)

df = pd.DataFrame(np.random.randn(6,4), index = dates, columns=list('ABCD'))

df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })

print(df.describe())
df['A'][0] = np.nan
print(df)
df = df.fillna(value=df.median()['A'])
print(df)
'''

df = pd.read_csv("iris.csv")

def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name:n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)

df2, targets = encode_target(df, "Name")

features = list(df2.columns[:4])

iris = load_iris()
'''
Tests with decision trees, knn

dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
dt.fit(df2[features],df2["Target"])

print(dt.predict_proba(iris.data[:1,:]))

clf = DecisionTreeRegressor()
clf = clf.fit(df2[features],df2["Target"])

print(clf.predict([[1,1,1,1]]))

knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data[:,:2],iris.target)

print(knn)
'''
X_sepal = iris.data[:, :2]  # Sepal
y_sepal = iris.target

X_petal = iris.data[:, 2:4] # Petal
y_petal = iris.target

def createScatter(range1, range2, n, y, x_label, y_label):
    x_min, x_max = range1.min() - 1, range1.max() + 1
    y_min, y_max = range2.min() - 1, range2.max() + 1
    plt.figure(n, figsize=(8, 6))
    plt.clf()
    
    # Plot the training points for Sepal
    plt.scatter(range1, range2, c=y, cmap=plt.cm.Set1,
            edgecolor='k')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

def createPlot(x,y,x_label,y_label,n):
    plt.figure(n, figsize=(8, 6))
    plt.clf()
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

createScatter(X_sepal[:,0], X_sepal[:,1], 1, y_sepal, "Sepal Length", "Sepal Width")

createScatter(X_petal[:,0], X_petal[:,1], 2, y_petal, "Petal Length", "Petal Width")

# split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(df2.ix[:,:4], df2["Target"], test_size=0.33, random_state=42)

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, Y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(Y_test, pred))

# CROSS VALIDATION TO AVOID OVERFITTING

# creating odd list of K for KNN
l = list(range(1,50))

# subsetting just the odd ones
neighbors = list(filter(lambda x: x % 2 != 0, l))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
createPlot(neighbors, MSE, "Number of neighbours K", "Misclassification Error",3)
createPlot(neighbors, cv_scores, "Number of neighbours K", "Classification Score",4)

#plt.show()

'''
KNN IMPLEMENTATION FROM SCRATCH
'''

# Training Block : Takes as input DATA (x) and TARGET (y) to output a LEARNED MODEL (h)
# Predict Block : Takes as input new and unseen observations, uses the LEARNED MODEL to give responses

#Training block memorizes training data
def train(x_train, y_train):
    return

#Predict block finds the distances between the points and outputs the K points closest to x_test 
def predict(x_train, x_test, y_train, k):
    distances = []
    targets = []
    
    for i in range(len(x_train)):
        distance = np.sqrt(np.sum(np.square(x_test - x_train[i, :])))
        distances.append([distance,i])

    distances = sorted(distances)

    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    return Counter(targets).most_common(1)[0][0]
    
#Test out prediction function
predict(X_train.as_matrix(), X_test.as_matrix(), Y_train.as_matrix(), 3)

#KNN main function, uses training block and finds the predictions for the points in x_test
def kNearestNeighbor(x_train, x_test, y_train, predictions, k):
    if k > len(X_train):
		raise ValueError

    train(x_train, y_train)

    for i in range(len(x_test)):
        predictions.append(predict(x_train, x_test[i, :], y_train, k))

predictions = []
#Test with k = 7
kNearestNeighbor(X_train.as_matrix(), X_test.as_matrix(), Y_train.as_matrix(), predictions, 7)
predictions = np.asarray(predictions)
accuracy = accuracy_score(Y_test.as_matrix(), predictions)
print(accuracy)
