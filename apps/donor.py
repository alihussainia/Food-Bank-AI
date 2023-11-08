import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pycaret.regression import load_model, predict_model

#with st.echo(code_location='below'):
def predict_rating(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    
    return predictions_data['Label'][0]

model = load_model("food_banks_model")

def app():
    st.header('Welcome to the Donors Section')
 
    st.markdown("👈 **Please select the Food Quantities (bags) that you wish to donate:**")
    st.write("Note: Leave the options to 0 that you don't want to donate now")

    #To provide multi-select with select all option
    
    selected_options = st.multiselect("Select one or more options:",
    ['SeaFood', 'Poultry', 'Bakery', 'Dairy','Fruites', 'Veggies'])
 
    all_options = st.checkbox("Select all options")
 
    if all_options:
        selected_options = ['SeaFood', 'Poultry', 'Bakery', 'Dairy','Frutis', 'Veggies']
    
    features={}
    for i in selected_options:
        v=i
        i = st.sidebar.number_input(label = str(i), value = 0, step=1)
        features[v]=i

    features_df  = pd.DataFrame(features, index=['Bags Selected for Acceptance'])

    if features:
        st.table(features_df)

    
    m = st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(247, 104, 104);
        color: rgb(255, 255, 255);
    }
    </style>""", unsafe_allow_html=True)


    if st.button('Find NGO'):

        prediction = predict_rating(model, features_df)
        
        st.write('Based on your donation level and food options, the most suitable NGO is '+ str(prediction))

