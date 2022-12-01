
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
import codecs
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from keras.optimizers import Adam

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from img_classification import slug_finder


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



def main_page():
    #st.markdown("# Sea Slug Classifier ")
    st.sidebar.markdown("# Opistobranch Classifier ")
    st.markdown("<h1 style='text-align: center;'>Opistobranch Classifier! ", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Upload your sea slug images in order to classify them! ", unsafe_allow_html=True)
    
    input_shape=(224,224,3)
    n_classes=5
    optm=Adam(learning_rate=0.0001)

    with st.spinner('Model is being loaded..'):
        vgg16_model = slug_finder(input_shape, n_classes, optimizer=optm, fine_tune=2)

    vgg16_model.load_weights('model_ignore/final_weights/VGG16_fine_tuning_yes_augmentation_yes_final.weights.best.hdf5')

    file= st.file_uploader('Drop your slug image here:', label_visibility="visible")

    st.set_option('deprecation.showfileUploaderEncoding', False)
    c1, c2= st.columns(2)
    if file is not None:

        im= Image.open(file)
        img= np.asarray(im)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_2 = cv2.resize(img_rgb, (224,224))

        #preprocess the image
        my_image = im_2.reshape((1, im_2.shape[0], im_2.shape[1], im_2.shape[2]))
        my_image = preprocess_input(my_image)


        c1.header('Input Image')
        c1.image(im)
        
        vgg_preds        = vgg16_model.predict(my_image)
        vgg_pred_classes = np.argmax(vgg_preds, axis=1)
        
        labels = ['Aplysiida', 'Cephalaspidea', 'Nudibranchia', 'Pleurobranchida', 'Runcinida']
        predicted_label = np.array(sorted(labels))[vgg_pred_classes]
        
        c2.header('Output')
        c2.subheader('Predicted class :')
        
        c2.write((str(predicted_label).replace("'", " ").replace("[", " ").replace("]", " ")))
    st.caption(""" Model classifies members of the orders: Aplysiidae, Cephalaspidea, Nudibranchia, Pleurobranchida, and Runcinida""") 

    



def page2():
    
    st.sidebar.markdown("# Model Information ")
    st.markdown("<h2 style='text-align: center;'>Model Information ", unsafe_allow_html=True)
    CM = Image.open("App/app_images/final_con.png")
    
    L_A = Image.open("App/app_images/final.jpg")
    choices= {
        'Confusion Matrix': CM,
        'Accuracy Loss': L_A
    }

    choice = st.sidebar.selectbox("Select Model Chart", list(choices.keys()))
    st.markdown("<h3 style='text-align: center;'> Model Test Accuracy: 90.45%", unsafe_allow_html=True)
    st.image(choices[choice], use_column_width=True, output_format= 'auto')




    

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
st.sidebar.image: st.sidebar.image("App/app_images/sidebar.png", use_column_width=True)