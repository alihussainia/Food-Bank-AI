import streamlit as st
from multiapp import MultiApp
from apps import home, donor, NGO # import your app modules here

app = MultiApp()

st.markdown("""
# Food Bank

Food Bank is a Counter Hunger initiative that is built in hackmaker's #buildwithai hackathon and aims to remove global hunger by connecting restuarants and bakeries to NGOs

""")

# Add all your application here
app.add_app("", home.app)
app.add_app("Donor", donor.app)
app.add_app("NGO", NGO.app)
# The main app
app.run()
