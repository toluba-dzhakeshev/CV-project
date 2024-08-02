import streamlit as st

st.title('Model Information Page')
st.header('Model Training Information')
num_epochs = 15

st.write(f"Number of Epochs: {num_epochs}")
st.write(f"Size of Train Dataset: {3500}, Validation: {600}, Test: {1000} images")
st.write("Test loss: 0.397679 | Test IoU: 0.730058 | Test accuracy: 0.8155")

acc_image_path = './images/tolu_acc.png'
iou_image_path = './images/tolu_iou.png'
loss_image_path = './images/tolu_loss.png'

st.header('Training Performance Plots')

st.image(acc_image_path, caption='Accuracy per Epoch', use_column_width=True)
st.image(iou_image_path, caption='IoU per Epoch', use_column_width=True)
st.image(loss_image_path, caption='Loss per Epoch', use_column_width=True)
