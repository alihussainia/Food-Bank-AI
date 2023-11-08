import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras import models
from datasets.sweden_food_banks import sweden_food_banks_dict 
from annotated_text import annotated_text

model = models.load_model('/mount/src/food-bank-ai/models/food_banks_classifier.keras')

def app():
    st.header('Welcome to the Donors Section')
 
    st.markdown("👈 **Please select the Food Quantities (bags) that you wish to donate:**")
    st.write("Note: Leave the options to 0 that you don't want to donate now")

    #To provide multi-select with select all option
    
    selected_options = st.multiselect("Select one or more options:",
    ['SeaFood', 'Poultry', 'Bakery', 'Dairy','Fruites', 'Veggies'])
    
    all_options = st.checkbox("Select all options")
 
    if all_options:
        selected_options = ['SeaFood', 'Poultry', 'Bakery', 'Dairy','Fruites', 'Veggies']
    
    features={'SeaFood':0,'Poultry':0,'Bakery':0,'Dairy':0,'Fruites':0,'Veggies':0}
    for option in selected_options:
        i = st.sidebar.number_input(label = str(option), value = 0, step=1, max_value=10,min_value=0)
        features[option]=i

    features_df  = pd.DataFrame(features, index=['Bags Selected for Donation'])

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
        features_lst = list(features.values())
        input_dict = np.array([features_lst])*1.0
        st.write(input_dict)
        predictions = model.predict(input_dict,verbose = 0)
        cls=np.argmax(predictions[0])
        prediction=sweden_food_banks_dict[cls]
        
        st.write('Based on your donation level and food options, the most suitable NGO is '+ prediction)

