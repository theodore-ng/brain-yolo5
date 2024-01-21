import streamlit as st
from clf import ImageClassifier
from PIL import Image
import requests
import json
from config import URL, HEADERS, DATA

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("ViT-GPT2 model demo by Tidrael")
st.write("")

uploaded_file = st.file_uploader("Upload an image")

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    # image_byte = uploaded_file.getvalue()
    response = requests.post(URL, headers=HEADERS, data=DATA, files={"image": image})
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    with st.spinner("Loading model, it could take a while..."):
        response = requests.post(URL, headers=HEADERS, data=DATA, files={"image": image})
        st.image(image, caption="Uploaded Image.", use_column_width=True)
    with st.spinner("Predict..."):
        response.raise_for_status()
    st.success(json.dumps(response.json(), indent=2))
