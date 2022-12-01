
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import codecs
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical


from img_classification import load_model


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
    """)
    with st.spinner('Model is being loaded..'):
        model= load_model()

    file= st.file_uploader('Drop your slug image here:', label_visibility="visible")

    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    c1, c2= st.columns(2)
    if file is not None:
        im= Image.open(file)
        img= np.asarray(im)
        image= cv2.resize(img,(224,224))
        img= tf.keras.applications.vgg16.preprocess_input(image)
        img= np.expand_dims(img, 0)
        c1.header('Input Image')
        c1.image(im)
        
      #load weights of the trained model.
        vgg_preds = model.predict(img)
        #y_classes = to_categorical(vgg_preds,5)
        vgg_pred_classes = np.argmax(vgg_preds, axis=1)
        
        labels = ['Aplysiida', 'Cephalaspidea', 'Nudibranchia', 'Pleurobranchida', 'Runcinida']
        predicted_label = np.array(sorted(labels))[vgg_pred_classes]
        c2.header('Output')
        c2.subheader('Predicted class :')
        
        c2.write(str(predicted_label))
    
        st.write("""#### Model classifies members of the orders: Aplysiidae, Cephalaspidea, Nudibranchia, Pleurobranchida, and Runcinida""")



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