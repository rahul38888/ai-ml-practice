import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler, StandardScaler

'''
    Load Data
'''
data = load_breast_cancer(as_frame=True)

X = data.data
y = data.target

def get_corr_col_lt(X, y, threshold=0.1):
    abs_corr = X.corrwith(y).abs()
    columns = list()
    values = list()
    for i in abs_corr.index:
        v = abs_corr[i]
        if v < threshold:
            columns.append(i)
            values.append(v)

    return pd.Series(values, index=columns)

'''
    Data plotting
'''
for col in get_corr_col_lt(X, y).index:
    sns.scatterplot(x=X[col], y=y)
    plt.show()


'''
    Data splitting
'''
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=0.25, random_state=42)
print(len(train_y), len(val_y), len(test_y))

min_max_scaler = MinMaxScaler()
train_X_p = min_max_scaler.fit_transform(train_X)
test_X_p = min_max_scaler.transform(test_X)
val_X_p = min_max_scaler.transform(val_X)

std_scaler = StandardScaler()
train_X_p = std_scaler.fit_transform(train_X)
test_X_p = std_scaler.transform(test_X)
val_X_p = std_scaler.transform(val_X)

def random_forest(scaler):
    '''
        Standardize data
            - MinMaxScalar in Bounded inputs, no outliers
            - StandardScaler with Gaussian (Normal) data, moderate outliers
            - RobustScalar Data with heavy outliers
    '''
    train_X_p = scaler.fit_transform(train_X)
    test_X_p = scaler.transform(test_X)
    val_X_p = scaler.transform(val_X)

    '''
        Random forrest classification
    '''
    rf = RandomForestClassifier(random_state=42)
    rf.fit(train_X, train_y)
    
    val_pred = rf.predict(val_X)
    acc = accuracy_score(val_y, val_pred)

    print(f"{scaler.__class__.__name__}: {acc}")

    '''
        MinMaxScaler: 0.9473684210526315
        StandardScaler: 0.9473684210526315
        RobustScaler: 0.9473684210526315
    '''