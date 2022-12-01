import keras
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2


def load_model():
  model= tf.keras.models.load_model('./model_ignore/VGG16_fine_tuning_2_no_augmentation')
  return model

 

def upload_predict(upload_image, model):
    
        size = (244,244)    
        image = ImageOps.fit(upload_image, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, dsize=(224, 224),interpolation=cv2.INTER_CUBIC)
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
    
        
        return prediction


def slug_finder(img, weights_file):
    # Load the model
    model = tf.keras.models.load_model('saved_model')

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS) 

    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    return np.argmax(prediction) # return position of the highest probability