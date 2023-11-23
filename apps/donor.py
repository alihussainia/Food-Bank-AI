import streamlit as st
import numpy as np
import pandas as pd
from datasets.sweden_food_banks import sweden_food_banks_dict 
import os
from trubrics import Trubrics
from joblib import load
    
model = load('models/food_banks.joblib')

if "my_form" not in st.session_state:
    st.session_state.my_form = None

trubrics = Trubrics(
    project="Food-Bank-AI",
    email=os.environ["TRUBRICS_EMAIL"],
    password=os.environ["TRUBRICS_PASSWORD"],
)
    
def app():
    global collector 
    st.header('Welcome to the Donors Section', divider='rainbow')
 
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
        i = st.sidebar.number_input(label = str(option), value = 0, step=1, max_value=8,min_value=0)
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

    findNGO = st.button('Find NGO')  
    if findNGO:
        features_lst = list(features.values())
        input_dict = np.array([features_lst])*1.0
        predictions = model.predict(input_dict)
        cls=np.argmax(predictions)
        prediction=sweden_food_banks_dict[cls]
        
        st.write('Based on your donation level and food options, the most suitable NGO is '+ prediction)

        st.header('Feedback Form', divider='rainbow')

        st.write('Please provide your feedback below :point_down:')
        
        with st.form(key="my_form"):
            st.write("Do you support Dark Theme for this App?")
            user_feedback = trubrics.log_feedback(
                component="default",
                feedback_type="thumbs",
                model="foodbank",
                prompt_id=None,
                # open_feedback_label="",
                #metadata={"input_features":features, "predicted_class": prediction},
                save_to_trubrics=True,
                align="center")


            # user_feedback1 = collector.st_feedback(
            #     component="DarkUIResponse",
            #     feedback_type="thumbs",
            #     model=model,
            #     metadata={"input_features":features, "predicted_class": prediction},
            #     save_to_trubrics=True,
            #     align="center")
            
            # st.session_state.count += 1
            # st.write("How do you feel about the App idea?")
            # user_feedback2 = collector.st_feedback(
            #     component="IdeaResponse",
            #     feedback_type="faces",
            #     model=model,
            #     metadata={"input_features":features, "predicted_class": prediction},
            #     save_to_trubrics=True,
            #     #key=st.session_state.count,
            #     align="center")
            
            # st.session_state.count += 1
            # st.write("[Optional] Provide any additional feedback about the App")
            # user_feedback3 = collector.st_feedback(
            #     component="FeedbackResponse",
            #     feedback_type="textbox",
            #     textbox_type="text-input",
            #     open_feedback_label="",
            #     model=model,
            #     metadata={"input_features":features, "predicted_class": prediction},
            #     #key=st.session_state.count,
            #     save_to_trubrics=True,
            #     align="center") 
            
            submitted = st.form_submit_button()
        
        if submitted and st.session_state.my_form:
            st.session_state.my_form = None
            st.toast("Thank You for Using Food Bank!")
            # st.write(user_feedback1)
            # st.write(user_feedback2)
            # st.write(user_feedback3)
            # st.session_state.feedback_key = 0
            #st.experimental_rerun()

