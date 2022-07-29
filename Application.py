import tensorflow
import streamlit as st

model = tensorflow.keras.models.load_model('saved_model/model5')

model.summary()

st.title("Dog Race Predicition")

data=st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

def predict_class():
    pred = model.predict_proba(data)
    st.write("The three most likely tags are, in decreasing probability order : ", pred)


if st.button("Predict"):
    predict_class()