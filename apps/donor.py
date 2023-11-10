import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras import models
from datasets.sweden_food_banks import sweden_food_banks_dict 
from annotated_text import annotated_text
from trubrics.integrations.streamlit import FeedbackCollector

model = models.load_model('/mount/src/food-bank-ai/models/food_banks_classifier.keras')


email=st.secrets.TRUBRICS_EMAIL,
password=st.secrets.TRUBRICS_PASSWORD,

@st.cache_data
def init_trubrics(email, password):
    collector = FeedbackCollector(
        email=email,
        password=password,
        project="foodbank")
    
collector = init_trubrics(email, password)

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
        predictions = model.predict(input_dict,verbose = 0)
        cls=np.argmax(predictions[0])
        prediction=sweden_food_banks_dict[cls]
        
        st.write('Based on your donation level and food options, the most suitable NGO is '+ prediction)

        st.write('Please provide your feedback below :point_down:')

        st.write("Do you support Dark Theme for this App?")
        user_feedback1 = collector.st_feedback(
        component="DarkUIResponse",
        feedback_type="thumbs",
        model=model,
        metadata={"input_features":features, "predicted_class": prediction},
        save_to_trubrics=True,
        align="flex-end") 

        st.write("What do you feel about the App idea?")
        user_feedback2 = collector.st_feedback(
        component="IdeaResponse",
        feedback_type="faces",
        model=model,
        metadata={"input_features":features, "predicted_class": prediction},
        save_to_trubrics=True,
        align="flex-end")

        st.write("[Optional] Feel free to provide any additional feedback about the App")
        user_feedback3 = collector.st_feedback(
        component="FeedbackResponse",
        feedback_type="textbox",
        textbox_type="text-input",
        model=model,
        metadata={"input_features":features, "predicted_class": prediction},
        save_to_trubrics=True,
        align="flex-end") 

        

