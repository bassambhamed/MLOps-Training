#import the needed librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

"""
This fonction serves to clean the data for trainining and it will split it into train and test needed for further steps
"""


def transform_data(df):
    #select the columns we need
    df = df[['trans_date_trans_time', 'category','amt', 'gender', 'zip', 'lat', 'long', 'dob', 'merch_lat', 'merch_long', 'is_fraud']]
    col_todate = ["trans_date_trans_time", "dob"]
    # transform specific cols to datetime type
    for col in col_todate:
        # convert trans_date_trans_time , dob to datetime
        df[col] = pd.to_datetime(df[col])
    #extract new cols
    # create new columns day,month,year
    df["year"] = df["trans_date_trans_time"].dt.year
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day"] = df["trans_date_trans_time"].dt.day
    # Extract hour,minute and second
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["minute"] = df["trans_date_trans_time"].dt.minute
    df["sec"] = df["trans_date_trans_time"].dt.second
    # Extract age of card holder column
    df['age'] = dt.date.today().year - pd.to_datetime(df['dob']).dt.year
    # drop unusefull columns
    df.drop(["dob", "trans_date_trans_time"], axis=1, inplace=True)
    # select numerical features
    num_features = df.select_dtypes(include=['integer']).columns.tolist()
    num_features.remove('is_fraud')
    # select categorical features
    categ_features = df.select_dtypes(include=['object']).columns.tolist()
    encode_dict = {  # Encoding dictionary
        'F': 0, 'M': 1}
    df['gender'] = df['gender'].map(encode_dict)
    dummy_cols = ['category']
    df = pd.get_dummies(df, columns=dummy_cols,dtype=float)
    X = df.drop('is_fraud', axis=1)  # Select features
    y = df['is_fraud']  # Target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)  # Split data 80/20
    #scaler = MinMaxScaler()  # Normalize train & test features
    #X_train[num_features] = scaler.fit_transform(X_train[num_features])
    #X_test[num_features] = scaler.transform(X_test[num_features])
    return X_train,X_test,y_train,y_test