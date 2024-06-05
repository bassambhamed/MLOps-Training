import requests
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import requests
import json
import os


# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
# st.set_option("deprecation.showfileUploaderEncoding", False)


# defines an h1 header
st.title("Fraud detector web app")
st.subheader("Enter details below")


#creating our form fields
with st.form("form1",clear_on_submit=False):
    trans_date_trans_time = st.text_input("Enter transaction datetime","2019-04-13 08:32:53")
    category = st.selectbox("Category of transaction", ('category_entertainment',
       'category_food_dining', 'category_gas_transport',
       'category_grocery_net', 'category_grocery_pos',
       'category_health_fitness', 'category_home', 'category_kids_pets',
       'category_misc_net', 'category_misc_pos', 'category_personal_care',
       'category_shopping_net', 'category_shopping_pos', 'category_travel')) ,
    amt = st.text_input("Transaction amount $","100") 
    gender = st.selectbox("Gender",('F',"M")) 
    zip = st.text_input("zip code","17060") 
    lat = st.text_input("Latitude Location of Credit Card Holder","36.0788") 
    long = st.text_input("Longitude Location of Credit Card Holder","-81.1781") 
    dob = st.date_input("When's the birthday of Credit Card Holder", datetime.date(1988, 3, 9)) 
    merch_lat = st.text_input("Latitude Location of Merchant","36.0788") 
    merch_long = st.text_input("Longitude Location of Merchant","-82.048315")
    dd={"trans_date_trans_time": trans_date_trans_time,
                "category" : category,
                "amt": amt,
                "gender": gender,
                "zip": zip,
                "lat": lat ,
                "long": long,
                "dob": dob.strftime("%y/%m/%d"),
                "merch_lat": merch_lat,
                "merch_long": merch_long}
    submit = st.form_submit_button("Submit this form")
    # res = requests.post("http://127.0.0.1:8000/predict", data=json.dumps(dd))
    res = requests.post("https://.../predict", data=json.dumps(dd))
    # res = requests.post("http://.../predict", data=json.dumps(dd))
    predictions = res.json().get("predictions")
    if predictions == [0] :
        st.text("It's not a fraud transaction 😃")
    else :

        st.text("Warning ! It's a fraud transaction 🚨")

st.subheader("Or Enter your historical transactions csv file")
# displays a file uploader widget
data = st.file_uploader("Choose a csv file")


# displays a button
if data is not None:
    file = {"file": data.getvalue()}
    #res = requests.post("http://127.0.0.1:8000/predict/csv", files=file)
    res = requests.post("https://.../predict/csv", files=file)
    #res = requests.post("http://.../predict/csv", files=file)
    predictions = res.json().get("predictions")
    st.text(predictions)

    
    