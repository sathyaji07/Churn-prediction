import streamlit as st
import pandas as pd
import numpy as np
#from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import os
from PIL import Image



# Sidebar navigation
r = st.sidebar.radio('Main Menu', ['Home', 'Customer Churn Prediction'])

# Home Page
if r == 'Home':
    st.title('BOOKSTORE - CUSTOMER CHURN PREDICTION 📚')
    st.subheader("Data has been processed from the publishing industry using ANN Deep Learning")
    st.markdown("*You can predict customer churn on the next page* 😎")
    st.image("c:/Users/DELL/OneDrive/Desktop/book.jpg")  

elif r == 'Customer Churn Prediction':
    image = Image.open("c:/Users/DELL/OneDrive/Desktop/book.jpg")
    resized_image = image.resize((700, 160))  # Width: 200px, Height: 150px
    st.image(resized_image) 

    left_column, right_column = st.columns(2)
    p1 = left_column.slider("How many days has the customer been buying books?", 0, 1200)
    p2 = right_column.slider("How many days has the customer not returned to the shop?", 0, 1200)

    if selected_date := left_column.date_input("Select an order date"):
        p3 = selected_date.day
        p4 = selected_date.month
        p5 = selected_date.year  

    p6 = right_column.slider("Price of the book bought?", 0, 500)
    p7 = left_column.number_input("How many times has the customer ordered from your shop?", min_value=0, max_value=500, value=10)

    # Load the model
    model = load_model('model.h5')  

    # Prepare input data
    input_data = np.array([[float(p6), float(p2), float(p1), float(p7), int(p3), int(p4), int(p5)]])

    if st.button('Predict'):
        pred = model.predict(input_data)
        predicted_class = np.argmax(pred, axis=1)[0]
        if predicted_class == 0:       
            st.success("The customer has not left the store")
        else:       
            st.warning("The customer has left the store. Try to give a discount of 5% to retain the customer")

