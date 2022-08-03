from PIL import Image
import numpy as np 
import streamlit as st 
from Number_plate_recognition_opencv import *
import cv2



#### Adding title
st.header("Automatic Number Plate Recognition")

# Function to load image
def load_image(photo):
    im = Image.open(photo)
    image = np.array(im)
    return image

# Uploading the File to the Page
uploadFile = st.file_uploader(label="Please Upload image here :", type=['jpg', 'png','jpeg'])

# Checking the Format of the page
if uploadFile is not None:
    photo = load_image(uploadFile)
    st.image(photo)
### Calling a function from Number_plate_recogntion_opencv.py and implemented into a clickabel button    
    if st.button("Recognize Plate Number"):
        plate, img, text_extract = num_extract(photo)
        st.image(img)
        st.subheader("The Actual Plate Number is: ")
        st.image(plate)
        st.subheader("The Predicted Plate Number is: ")
        st.write(text_extract)

