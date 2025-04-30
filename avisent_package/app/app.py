import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
from PIL import Image


# ✅ Set page config
st.set_page_config(page_title="AviSent", page_icon="🛫", layout="wide")

# ✅ Set background image from local file
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set the background
set_background("Riyadh20Image%2012.jpg")

# ✅ Display Logo
logo = Image.open("gg-0١.png")
st.image(logo, width=200)


with open('model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_xgb.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('model_NLP_newv2.pkl', 'rb') as f:
    nlp = pickle.load(f)


# ✅ Sidebar user input
with st.sidebar:
    st.markdown("### 📜 <span style='color:#0E76A8;'>User Input Features</span>", unsafe_allow_html=True)
    Overall_Rating = st.slider('Overall Rating (1-10)', 1, 10)
    Seat_Comfort = st.slider('Seat Comfort (1-5)', 1, 5)
    Cabin_Staff_Service = st.slider('Cabin Staff Service (1-5)', 1, 5)
    Food_Beverages = st.slider('Food & Beverages (1-5)', 1, 5)
    Ground_Service = st.slider('Ground Service (1-5)', 1, 5)
    Type_Of_Traveller = st.selectbox('Type of Traveller', ['Business', 'Couple Leisure', 'Family Leisure', 'Solo Leisure'])
    Seat_Type = st.selectbox('Seat Type', ['Economy Class', 'Premium Economy', 'Business Class', 'First Class'])

# Encode categorical features
traveller_encoding = {
    'Business': [1, 0, 0, 0],
    'Couple Leisure': [0, 1, 0, 0],
    'Family Leisure': [0, 0, 1, 0],
    'Solo Leisure': [0, 0, 0, 1]
}
seat_encoding = {
    'Economy Class': 0,
    'Premium Economy': 1,
    'Business Class': 2,
    'First Class': 3
}
traveller_encoded = traveller_encoding[Type_Of_Traveller]
seat_encoded = seat_encoding[Seat_Type]

features = np.array([[Overall_Rating, Seat_Comfort, Cabin_Staff_Service, Food_Beverages, Ground_Service,
                      *traveller_encoded, seat_encoded]])

# ✅ App title and description with white text
st.markdown("""
    <h1 style='color:white; font-size: 48px; font-weight: bold; margin-bottom: 10px;'>
        AviSent — Airline Recommendation Predictor
    </h1>
""", unsafe_allow_html=True)
st.markdown("<h4 style='color:white;'>Based on your flight experience, would you recommend this airline?</h4>", unsafe_allow_html=True)
st.markdown("<h5 style='color:white;'>Please share your thoughts about the flight below:</h5>", unsafe_allow_html=True)

# ✅ Review input
review = st.text_input("")

# ✅ Submit button
if st.button('✅ Submit'):
    st.markdown("<div style='color:white; font-size:20px;'>✅ Thank you for your time!</div>", unsafe_allow_html=True)


# ✅ Predict button
if st.button('🔍 Predict'):
    scaled_input = features.copy()
    scaled_input[:, :5] = scaler.transform(scaled_input[:, :5])
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)

    if prediction[0] == 1:
        st.markdown(f"<div style='color:white; font-size:18px;'>Classifier: Customer is <b>LIKELY</b> to recommend the airline (Confidence: {prediction_proba[0][1]*100:.2f}%)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:white; font-size:18px;'>Classifier: Customer is <b>NOT likely</b> to recommend the airline (Confidence: {prediction_proba[0][0]*100:.2f}%)</div>", unsafe_allow_html=True)


    if review.strip():
        sentiment = nlp.predict(pd.Series([review]))[0]
        if sentiment == 1:
            st.markdown("<div style='color:white; font-size:18px;'>Sentiment: Customer is <b>LIKELY</b> to recommend the airline</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color:white; font-size:18px;'>Sentiment: Customer is <b>NOT likely</b> to recommend the airline</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:white; font-size:16px;'>ℹ️ No review provided for sentiment analysis.</div>", unsafe_allow_html=True)
