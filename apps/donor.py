import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import models
from dataset.sweden_job_banks import sweden_food_banks_dict 

model = tensorflow.keras.models.load_model('/mount/src/food-bank-ai/models/food_banks_classifier.keras')

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
    
    features={'SeaFood':0,'Poultry':0,'Bakery':0,'Dairy':0,'Frutis':0,'Veggies':0}
    for i in selected_options:
        v=i
        i = st.sidebar.selectbox(label = str(i), (1,2,3,4,5,6,7,8,9,10))
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

        input_dict = np.array([list(features_df.values())])*1.0
        predictions = model.predict(input_dict,verbose = 0)
        cls=np.argmax(predictions[0])
        prediction=job_bank_dict[cls]
        
        st.write('Based on your donation level and food options, the most suitable NGO is '+ prediction)

