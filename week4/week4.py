

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression

def main():
    df1 = csv_to_df("df1.csv")

    classification_scatterplot(df1, "fig1")

    x = get_polynomials(df1)
    y = df1[['y']]

    
    model = LogisticRegression()
    model.fit(x[[]], y)

    
def csv_to_df(filename):
    """
    convert data in CSV format to DataFile.

    note: only works for data with
    feature values x1,x2
    target value y.
    """
    data = pd.read_csv(filename, header=0, names=['x1','x2','y'])
    return data

def classification_scatterplot(data, figname):
    """
    visualise the data placing a marker on a 2D plot for each 
    pair of feature values. 
    x-axis is first feature.
    y-axis is second feature.
    + marker when the target value is +1
    o when the target is -1. 
    """
    plus = data[data['y']>0]
    minus = data[data['y']<0]

    plt.figure()

    plt.scatter(plus['x1'], plus['x2'], marker='P', color='green', label="y = +1")
    plt.scatter(minus['x1'], minus['x2'], marker='o', color='red', label="y = -1")

    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.legend()

    plt.savefig(figname)

def insert_polynomials(data):
    """
    augment the two features in the datafile 
    with polynomial features.

    Parameters
    ----------
    data : DataFile.

    Returns
    -------
    DataFile
        augmented data.
    """
    targets = data[['y']]
    features = data[['x1','x2']]
    
    poly = PolynomialFeatures(degree=5, include_bias=False)
    features = poly.fit_transform(features)

    data = pd.DataFrame(features, columns=poly.get_feature_names_out())
    data.insert(data.shape[1], 'y', targets)

    return data

def get_polynomials(data):
    """
    return augmented features
    """
    features = data[['x1','x2']]
    
    poly = PolynomialFeatures(degree=5, include_bias=False)
    features = poly.fit_transform(features)

    features = pd.DataFrame(features, columns=poly.get_feature_names_out())
    
    return features


if __name__ == "__main__":
    main()