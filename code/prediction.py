import math
from numpy import concatenate
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib
import string
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import joblib
import sys
import csv

title = ""

def load_csv_with_single_region(region):
    output = []
    isolated = []

    with open("dataset/raw_russia_demography.csv", 'r') as input:
        reader = csv.reader(input, delimiter = ',')
        all = []
        row = next(reader)
        all.append(row)
        isolated.append(row)
        for k, row in enumerate(reader):
            if (row[1] == 'region' or row[1] == region):
                isolated.append(row)
            all.append([str(k+1)] + row)
        output = all
    dataframe = pd.DataFrame(isolated) 
    return dataframe

def main(region):
    global title
    data = load_csv_with_single_region(region)
    title = region
    ReframingandSpliting(encoding(handling_data(data)), 1)
    return

def handling_data(data):
    try:
        data = data.drop(data.index[:1])
        data = data.drop(2, axis=1)
        data = data.drop(4, axis=1)
        data = data.drop(5, axis=1)
        data = data.drop(6, axis=1)
        data = data.sort_values(by = 1, ascending = True)
        return(data)
    except OSError as e:
        return(e)

def encoding(data):
    data = data.drop(1, axis=1)
    data.columns = ['ID', 'birth rate']
    values = data.values
    encoder = LabelEncoder()
    values[:,1] = encoder.fit_transform(values[:,1])
    return values

def ReframingandSpliting(data, road):
    reframed = series_to_supervised(data, 1, 1)
    road = 1
    if (road == "1"):
        reframed.drop(reframed.columns[[-3,-1]], axis=1, inplace=True)

    values = reframed.values
    scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
    
    scaled_features = scaler.fit_transform(values[:,:-1])
    scaled_label = scaler.fit_transform(values[:,-1].reshape(-1,1))
    values = np.column_stack((scaled_features, scaled_label))

    train = values[:20, :]
    test = values[20:, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    return fitandpredictforpredict(train_X, train_y, test_X, test_y, scaler, road)

def fitandpredictforpredict(train_X, train_y, test_X, test_y, scaler, road):
    x = train_X
    y = train_y

    regr = SVR(C = 2.0, epsilon = 0.1, kernel = 'rbf', gamma = 0.5, 
                tol = 0.001, verbose=2, shrinking=True, max_iter = -1)
    
    regr.fit(x, y)
    data_pred = regr.predict(x)
    y_pred = scaler.inverse_transform(data_pred.reshape(-1,1))
    y_inv = scaler.inverse_transform(y.reshape(-1,1))
    mse = mean_squared_error(y_inv, y_pred)
    draw_prediction(y_pred[:10], y_inv[:10])
    return y_pred[:1,]

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
	    agg.dropna(inplace=True)
    return agg

def draw_prediction(preds, actual):
    fig, ax = plt.subplots(figsize=(15,8))
    ax.plot(preds, color='red', label='Predicted data')
    ax.plot(actual, color='green', label='Previous test data')
    fig.canvas.set_window_title(title)
    fig.suptitle(title, size=15)
    plt.xlabel('Years', fontsize=15)
    plt.ylabel('Births', fontsize=15)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
