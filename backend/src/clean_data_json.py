
import pandas as pd
import pickle
import datetime as dt
"""
This function serves to clean the incoming new data in production when it is JSON format
"""

training_cols = ['amt', 'gender', 'zip', 'lat', 'long', 'merch_lat', 'merch_long',
       'year', 'month', 'day', 'hour', 'minute', 'sec', 'age',
       'category_entertainment', 'category_food_dining',
       'category_gas_transport', 'category_grocery_net',
       'category_grocery_pos', 'category_health_fitness', 'category_home',
       'category_kids_pets', 'category_misc_net', 'category_misc_pos',
       'category_personal_care', 'category_shopping_net',
       'category_shopping_pos', 'category_travel']

def fix_missing_cols(training_cols, new_data):
    missing_cols = set(training_cols) - set(new_data.columns)
     # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        new_data[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    new_data = new_data[training_cols]
    return new_data




def clean_data_json(df):
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
    # select categorical features
    categ_features = df.select_dtypes(include=['object']).columns.tolist()
    encode_dict = {  # Encoding dictionary
        'F': 0, 'M': 1}
    df['gender'] = df['gender'].map(encode_dict)
    dummy_cols = ['category']
    df = pd.get_dummies(df, columns=dummy_cols,dtype=float)
    df = fix_missing_cols(training_cols,df)
    return df