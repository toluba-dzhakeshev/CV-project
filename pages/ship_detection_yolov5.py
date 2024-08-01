import streamlit as st
import torch
from PIL import Image

from torchvision import transforms as T

st.title('YOLOv5')

@st.cache_resource
def get_model(conf):
    model = torch.hub.load(
        # будем работать с локальной моделью в текущей папке
        repo_or_dir = './yolov5/',
        model = 'custom', 
        path='best.pt', 
        source='local' 
        )
    model.eval()
    model.conf = conf
    print('Model loaded')
    return model

with st.sidebar:
    t = st.slider('Model conf', 0., 1., .1)

with st.spinner():
    model = get_model(t)


uploaded_file = st.file_uploader('Upload image', type=['jpeg', 'jpg'])
results=None
lcol, rcol = st.columns(2)
with lcol:
    if uploaded_file:
        img = Image.open(uploaded_file)
        results = model(img)
        st.image(img)

if results:
    with rcol:
        st.image(results.render())