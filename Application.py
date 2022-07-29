import tensorflow
import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image

model = tensorflow.keras.models.load_model("C:/Users/antoi/Dropbox/PC/Documents/GitHub/Projet-6/model5.h5")

model.summary()

probability_model = tensorflow.keras.Sequential([model,tensorflow.keras.layers.Softmax()])
class_names=np.load("C:/Users/antoi/Dropbox/PC/Documents/GitHub/Projet-6/class_name.npy",allow_pickle=True)
st.title("Dog Race Predicition")
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

def predict_class():
    image = Image.open(uploaded_file)
    img = np.array(image)
    st.text(img.shape)
    img=np.asarray([cv.resize(img,(100,100))])
    pred = probability_model.predict(img)
    st.write("This dog is most likely a : ", class_names[np.argmax(pred)])


if st.button("Predict"):
    predict_class()