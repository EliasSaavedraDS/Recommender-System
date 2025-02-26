import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def train_isolation_forest(data_copy, features, n_estimators=1000, bootstrap=False, max_samples='auto'):
    '''This function takes as input the DataFrame and corresponding features and fit Isolation forest'''

    X_train = data_copy[features]

    clf=IsolationForest(n_estimators=n_estimators, max_samples=max_samples, \
       bootstrap=bootstrap, n_jobs=-1, random_state=13, verbose=0)

    clf.fit(X_train)

    return clf

def get_anomaly_and_score(data_copy, features, clf):
    '''This function takes as input DataFrame, trained model for outliers and imputer used in training
    and update columns with anomaly (1 being normal and -1 being outlier)'''

    X_train = data_copy[features]

    pred = clf.predict(X_train)

    data_copy['anomaly']=pred

    pred_score = clf.score_samples(X_train)

    data_copy['score_anomaly']=pred_score

    return data_copy

def get_outliers_index(new_data,mode = 'normal', threshold = -0.5 , percent = 0.5):
    '''This function takes as input a DataFrame and return indexes of outliers and not outliers.
    Three modes: normal mode that output all outliers, threshold mode that ouput based on a threshold
    and percent that output based on a percent of outliers'''

    if mode == 'normal':

        outliers=new_data.loc[new_data['anomaly']==-1]
        outlier_index=list(outliers.reset_index(drop=True).index)
        clean=new_data.loc[new_data['anomaly']==1]
        clean_index=list(clean.reset_index(drop=True).index)

    elif mode == 'threshold':

        outliers=new_data.loc[new_data['score_anomaly']<threshold]
        outlier_index=list(outliers.index)
        clean=new_data.loc[new_data['score_anomaly']>=threshold]
        clean_index=list(clean.index)

    elif mode == 'percent':

        threshold = new_data.sort_values(by='score_anomaly')[:int(new_data.shape[0]*(percent/100))]['score_anomaly'].values[-1]
        outliers=new_data.loc[new_data['score_anomaly']<threshold]
        outlier_index=list(outliers.index)
        clean=new_data.loc[new_data['score_anomaly']>=threshold]
        clean_index=list(clean.index)

    return outlier_index, clean_index


def plot_anomaly(new_data, features, outlier_index, clean_index, mode = '3D'):
    '''This function takes dataFrame as input, with features of interest and indexes of outliers
    and plot the 3D or 2D (depending on the mode) PCA plot of outlier vs normal
    '''

    if mode == '3D':
        scaler = StandardScaler()
        #normalize the features
        X_train = new_data[features]

        X = scaler.fit_transform(X_train)

        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlabel("x_composite_3")

        # Plot the compressed data points
        ax.scatter(X[:, 0], X[:, 1], zs=X[:, 2], s=4, lw=1, label="inliers",c="green")

        # Plot x's for the ground truth outliers
        ax.scatter(X[outlier_index,0],X[outlier_index,1], X[outlier_index,2],
                   lw=2, s=60, marker="x", c="red", label="outliers")
        ax.legend()
        plt.show()

    elif mode == '2D':

        plt.figure(figsize=(10,8))
       
        scaler = StandardScaler()
        #normalize the features
        X_train = new_data[features]
        X = scaler.fit_transform(X_train)
        #Reduce dimension with PCA
        

        # Plot the compressed data points
        plt.scatter(X[clean_index,0],X[clean_index,1],label='normal points')
        plt.scatter(X[outlier_index,0],X[outlier_index,1],c='red', label='predicted outliers')
        plt.legend(loc="upper right")
        plt.show()

    return