import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn import neighbors

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

dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
dt.fit(df2[features],df2["Target"])

iris = load_iris()

print(dt.predict_proba(iris.data[:1,:]))

clf = DecisionTreeRegressor()
clf = clf.fit(df2[features],df2["Target"])
print(clf.predict([[1,1,1,1]]))

knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data[:,:2],iris.target)

print(knn)

X = iris.data[:, :4]  # All four features
y = iris.target

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

plt.figure(2, figsize=(8, 6))
plt.clf()

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1,
            edgecolor='k')
plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.Set1,
            edgecolor='m')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()