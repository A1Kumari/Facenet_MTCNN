import streamlit as st
import cv2
import base64
from detection import *

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://images.pexels.com/photos/1054218/pexels-photo-1054218.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2");
    background-size: cover;
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

st.html("<h1><center>Face Detection Web App</h1></center>")
vis = True
db_mode = st.checkbox("Add to data base")
if db_mode:
    vis = False
if not vis:
    st.html("<h3><center>Storing Image to Database</h3></center>")
input_method = st.checkbox("Camera Mode")
cam_inp , file_inp = None , None
if input_method:
    cam_inp = st.camera_input(label="Say Cheese!")
else:
    file_inp = st.file_uploader("Upload a Photo!")

if cam_inp != None:
    file = cam_inp
else:
    file = file_inp

if file and vis:
    min_conf = st.number_input("Enter the desired confidence value:" , min_value=0.0 , max_value=1.0, step=0.10, disabled= not vis , value=0.65)
    if min_conf:
        img , labels = detect_and_fetch(file , min_confidence= min_conf)
        st.image(img , caption='Detections' , use_column_width=True)
        st.header('Detected:')
        labels = set(labels)
        for i in labels:
            if i is not "Unknown":
                st.text(i)
else:
    name = st.text_input("Write the Name of Person" , disabled= vis)
    if name:
        process = write_and_upsert(file , name)
        if process:
            st.text("Uploaded Sucessfully!")

