import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras import models
from datasets.sweden_food_banks import sweden_food_banks_dict 

model = models.load_model('/mount/src/food-bank-ai/models/food_banks_classifier.keras')

def app():
    st.header('Welcome to the NGO Section')
 
    st.markdown("👈 **Please select the Food Quantities (bags) that you wish to accept:**")
    st.write("Note: Leave the options to 0 that you don't want to accept now")
    
    #To provide multi-select with select all option
    
    selected_options = st.multiselect("Select one or more options:",
    ['SeaFood', 'Poultry', 'Bakery', 'Dairy','Fruites', 'Veggies'])
 
    all_options = st.checkbox("Select all options")
 
    if all_options:
        selected_options = ['SeaFood', 'Poultry', 'Bakery', 'Dairy','Frutis', 'Veggies']
    
    features={}
    for i in selected_options:
        v=i
        i = st.sidebar.number_input(label = str(i), value = 0, step=1, max_value=10)
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
 
 
    if st.button('Find Donor'):
        
        
        features_lst = list(features.values())
        input_dict = np.array([features_lst])*1.0
        predictions = model.predict(input_dict,verbose = 0)
        cls=np.argmax(predictions[0])
        prediction=sweden_food_banks_dict[cls]
        
        
        st.write('Based on your acceptance level and food options, the most suitable donor is '+ prediction)
 
 
 
 
 
 
 
 
 
 
 
 
 
 

