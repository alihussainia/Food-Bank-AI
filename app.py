import streamlit as st
from multiapp import MultiApp
from apps import home, donor, NGO # import your app modules here

app = MultiApp()

st.markdown("""
# Food Bank

Food Bank is a Counter Hunger initiative with an aim to remove food wastage and proverty driven hunger issues by connecting food donors with food banks.

""")

# Add all your application here
app.add_app("", home.app)
app.add_app("Donor", donor.app)
app.add_app("NGO", NGO.app)
# The main app
app.run()
