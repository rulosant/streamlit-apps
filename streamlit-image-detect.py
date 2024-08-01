import streamlit as st
import numpy as np
import cv2 as cv2
from ultralytics import YOLO
import os

from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

st.set_page_config(
  page_title="YOLO Model Tester",
  page_icon="🚀"
)

def run(source, conf_thres=0.25):
    results = model.predict(source, conf=conf_thres)
    return results[0].plot()

st.title('YOLO Model Tester')

st.markdown('This is an application for object detection using YOLO')



# Load the YOLO model
model = YOLO('../yolov8m.pt')  # Adjust the path to your YOLO model weights



# Use a List instead of a Generator
available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolov8")]

# Scan the current folder for .pt files
current_folder_models = [f.replace(".pt", "").replace("yolo", "YOLO") for f in os.listdir('.') if f.endswith('.pt')]

# Add current folder .pt files to the available models list if not already included
unique_models = [model for model in current_folder_models if model not in available_models]
available_models = unique_models + available_models

selected_model = st.sidebar.selectbox("Model", available_models)
with st.spinner("Model is downloading..."):
    model = YOLO(f"{selected_model.lower()}.pt")  # Load the YOLO model
    class_names = list(model.names.values())  # Convert dictionary to list of class names
st.sidebar.success("Model loaded successfully!")


img_files = st.sidebar.file_uploader(label="Choose an image files",
                 type=['png', 'jpg', 'jpeg'],
                 accept_multiple_files=True)



for n, img_file_buffer in enumerate(img_files):
  if img_file_buffer is not None:
    # we'll do this later
    # 1) image file buffer will converted to cv2 image
    # 2) pass image to the model to get the detection result
    # 3) show result image using st.image()
    
    # function to convert file buffer to cv2 image
    def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
        img_stream.seek(0)
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)

    # 1) image file buffer will converted to cv2 image
    open_cv_image = create_opencv_image_from_stringio(img_file_buffer)

    # 2) pass image to the model to get the detection result
    im0 = run(source=open_cv_image, \
    conf_thres=0.4)

    # 3) show result image using st.image()
    if im0 is not None:
        st.image(im0, channels="BGR", \
        caption=f'Detection Results ({n+1}/{len(img_files)})')


st.markdown("""
  <p style='text-align: center; font-size:16px; margin-top: 32px'>
    AwesomePython @2020
  </p>
""", unsafe_allow_html=True)
