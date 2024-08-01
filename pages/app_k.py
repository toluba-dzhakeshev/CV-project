import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
from torchvision.io import read_image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import requests
from model_utils_k import ResNet18ForLocalization,load_image_from_url, predict_and_display, load_history, plot_history, load_yolov5_model, save_image_to_bytes


st.title('Локализация огурчиков')

model_option = st.selectbox('Выберите модель', ['ResNet18', 'YOLOv5'])

if model_option == 'ResNet18':
    
    st.subheader('Dataset Information')
    history = load_history('pages/training_history.json')
    buf = plot_history(history)
    st. image(buf)
    
    dataset_info = {
        'Number of classes': 3,
        'Number of training images': 150,
        'Number of validation images': 30,
        'Training time, sec': 52 
    }
    st.write(dataset_info)
    option = st.selectbox('Выберите способ загрузки', ['From URL', 'From File'])

    if option == 'From URL':
        url = st.text_input('Image URL', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWgudRC2zhtWViDI0JkDt9fS1GTPQ-exJYEg&s')
        if st.button('Classify'):
            if url:
                image = load_image_from_url(url)
                predict_and_display(image)
            else:
                st.error("Please enter a valid URL.")

    elif option == 'From File':
        uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                predict_and_display(image)
                

if model_option == 'YOLOv5':
    
    img1 = Image.open('images/confusion_matrix_K.png')
    img2 = Image.open('images/PR_curve_K.png')
    img3 = Image.open('images/results_K.png')
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img1, caption='Confusion Matrix', use_column_width=True)
    with col2:
        st.image(img2, caption='PR Curve', use_column_width=True)
    with col3:
        st.image(img3, caption='Metrics', use_column_width=True)
        
    dataset_info = {
        'Number of classes': 3,
        'Number of training images': 393,
        'Number of validation images': 36,
        'Training time, m': 15 
    }
    st.write(dataset_info)
    
    model_path = 'models/best_k.pt'
    model = load_yolov5_model(model_path)

    option = st.selectbox('Выбери способ загрузки изображения', ['From URL', 'From File'])

    if option == 'From URL':
        url = st.text_input('Image URL', 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRWgudRC2zhtWViDI0JkDt9fS1GTPQ-exJYEg&s')
        if st.button('Classify'):
            if url:
                image = load_image_from_url(url)
                model.eval()
                with torch.no_grad():
                    results = model(image)
                
                results_img = results.render()[0]
                buf = save_image_to_bytes(Image.fromarray(results_img))
                
                st.image(buf, caption='YOLOv5 Prediction')
            else:
                st.error("Please enter a valid URL.")

    elif option == 'From File':
        uploaded_files = st.file_uploader("Choose images...", accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert("RGB")
                model.eval()
                with torch.no_grad():
                    results = model(image)
                
                # Convert results to PIL Image
                results_img = results.render()[0]  # Get the first image with results
                buf = save_image_to_bytes(Image.fromarray(results_img))
                
                st.image(buf, caption='YOLOv5 Prediction')
    