import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier

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
