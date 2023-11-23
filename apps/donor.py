import streamlit as st
import numpy as np
import pandas as pd
from datasets.sweden_food_banks import sweden_food_banks_dict 
from trubrics.integrations.streamlit import FeedbackCollector
from joblib import load
    
model = load('models/food_banks.joblib')


if 'count' not in st.session_state:
	st.session_state.count = 0

submitted1 = None
collector = FeedbackCollector(
    email=st.secrets.TRUBRICS_EMAIL,
    password=st.secrets.TRUBRICS_PASSWORD,
    project="Food-Bank-AI")
    
def app(): 
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
        st.session_state.count += 1
        features_lst = list(features.values())
        input_dict = np.array([features_lst])*1.0
        predictions = model.predict(input_dict)
        cls=np.argmax(predictions)
        prediction=sweden_food_banks_dict[cls]
        
        st.write('Based on your donation level and food options, the most suitable NGO is '+ prediction)

        st.header('Feedback Form', divider='rainbow')

        st.write('Please provide your feedback below :point_down:')
        
        with st.form(key='count'):
            st.write("Do you support Dark Theme for this App?")
            user_feedback1 = collector.st_feedback(
                component="DarkUIResponse",
                feedback_type="thumbs",
                model=model,
                metadata={"input_features":features, "predicted_class": prediction},
                save_to_trubrics=True,
                key=st.session_state.count,
                align="center")
            if user_feedback1:
                st.session_state.count += 1
		st.write("How do you feel about the App idea?")
		user_feedback2 = collector.st_feedback(
			component="IdeaResponse",
			feedback_type="faces",
			model=model,
			metadata={"input_features":features, "predicted_class": prediction},
			save_to_trubrics=True,
			key=st.session_state.count,
			align="center")
		            
                if user_feedback2:
                    st.session_state.count += 1
                    st.write("[Optional] Provide any additional feedback about the App")
                    user_feedback3 = collector.st_feedback(
                        component="FeedbackResponse",
                        feedback_type="textbox",
                        textbox_type="text-input",
                        open_feedback_label="",
                        model=model,
                        metadata={"input_features":features, "predicted_class": prediction},
                        key=st.session_state.count,
                        save_to_trubrics=True,
                        align="center") 
            
            submitted1 = st.form_submit_button('Submit Feedback')
        
        if submitted1:
            st.toast("Thank You for Using Food Bank!")
            # st.session_state.feedback_key = 0
            # st.session_state.key_1 = 1
            # st.session_state.key_2 = 2
            # st.session_state.key_3 = 3
            #st.experimental_rerun()

