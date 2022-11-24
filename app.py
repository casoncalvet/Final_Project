
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import codecs

st.set_page_config(
     page_title="SeaSlugSearcher",
     page_icon="app_images/icon.png",
     layout="wide",
     initial_sidebar_state="expanded",
     menu_items={
         'Get Help': 'https://www.seaslugsearcher.com/help',
         'Report a bug': "https://www.seaslugsearcher.com/bug",
         'About': "# Classify your slugs! Automated taxonomy!"
     }
 )

cover = Image.open("app_images/cover.jpg")
st.image(cover, use_column_width=True)
st.write("""
# Sea Slug Order Identifyer! 
##     Upload your sea slug images in order to classify them! 
""")
st.video("https://www.youtube.com/watch?v=F7V8DRfZBQI")

#components.html(pedro,height=550,scrolling=True)

