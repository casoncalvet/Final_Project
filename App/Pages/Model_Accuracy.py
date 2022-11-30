import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import codecs

st.set_page_config(
     page_title="Model Accuracy",
     page_icon="App/app_images/icon.png",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'About': "# Classify your slugs! Automated taxonomy!"
     }
 )

st.write("""
# Model Information 
""")

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()