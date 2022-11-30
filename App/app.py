
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import codecs

#from img_classification import slug_finder


st.set_page_config(
    page_title="SeaSlugSearcher",
    page_icon="App/app_images/icon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# Classify your slugs! Automated taxonomy!"
    }
)

cover = Image.open("App/app_images/cova.jpeg")
st.image(cover, use_column_width=True)



#components.html(pedro,height=550,scrolling=True)



def main_page():
    #st.markdown("# Sea Slug Classifier ")
    st.sidebar.markdown("# Opistobranch Classifier ")
    st.write("""
    # Opistobranch Classifier! 
    ##     Upload your sea slug images in order to classify them!
    #### Model classifies members of the orders: Aplysiidae, Cephalaspidea, Nudibranchia, Pleurobranchida, and Runcinida
    """)
    #uploaded_file = st.file_uploader("Show us your slug!", type="jpg")
    #if uploaded_file is not None:
    #    image = Image.open(uploaded_file)
    #    st.image(image, caption='Uploaded image', use_column_width=True)
    #    st.write("")
    #    st.write("Classifying...")
    #    label = slug_finder(image, 'model.h5')

def page2():
    st.markdown("# Model Information ")
    st.sidebar.markdown("# Model Information ")

    

def page3():
    st.markdown("# Opistobranch Resources ")
    st.sidebar.markdown("# Resources ")
    st.write("""
    ###     More Resources for Classification: 
    """)
    st.markdown("- Opisitobranch Taxonomy and Gallery: [OPK Opistobranquis](https://opistobranquis.info/en/)")
    st.markdown("- Image Classification available at: [iNaturalist](https://www.inaturalist.org/observations?taxon_id=551391)")
    st.markdown("- Species List and Classification: [SeaSlugForum](http://www.seaslugforum.net/specieslist.htm)")
    

    st.markdown('''
    <style>
    [data-testid="stMarkdownContainer"] ul{
        padding-left:40px;
    }
    </style>
    ''', unsafe_allow_html=True)


page_names_to_funcs = {
    "Opistobranch Classifier": main_page,
    "Model Information": page2,
    "More Resources": page3,
}

selected_page = st.sidebar.selectbox("Menu", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
st.sidebar.image: st.sidebar.image("App/app_images/side.png", use_column_width=True)