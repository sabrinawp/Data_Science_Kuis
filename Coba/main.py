import pandas as pd
import numpy as np
from numpy import genfromtxt
from classifier import *
from sklearn.metrics import accuracy_score
from feature_selection import *

if __name__ == '__main__':
    
    # dataset
    df = genfromtxt('datatrainquiz.csv', delimiter=',', skip_header=1)
    X = df[:, 0:7]
    y = df[:, 7:8]
    yy = []
    for i in range(len(y)):
        # print(y[i][0])
        yy.append(y[i][0])
    y = yy
    # print("X = ",X)
    # print("y = ",y)
    df = pd.read_excel(open('datatesting.xlsx', 'rb'))
    X_test = pd.DataFrame (df, columns=(['A', 'B', 'C', 'D', 'E', 'F', 'G']))
    X_test=np.array(X_test)
    
    # fitur selection
    select = feature_selection(X,y)
    X = select.selectKbest_regression(7)
    # balancing
    # classifier
    LinearRegression = classLinearRegression(X,y)
    LinearRegression.model()
    y_pred = LinearRegression.predict(X_test)
    print(y_pred)