import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open('/avisent_package/app/model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

with open('/avisent_package/app/scaler_xgb.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('/avisent_package/app/model_NLP_newv2.pkl', 'rb') as f:
    nlp = pickle.load(f)

def user_input_features():
    st.sidebar.header('Input Airline Review Features')

    Overall_Rating = st.sidebar.slider('Overall Rating (1-10)', 1, 10)
    Seat_Comfort = st.sidebar.slider('Seat Comfort (1-5)', 1, 5)
    Cabin_Staff_Service = st.sidebar.slider('Cabin Staff Service (1-5)', 1, 5,)
    Food_Beverages = st.sidebar.slider('Food & Beverages (1-5)', 1, 5,)
    Ground_Service = st.sidebar.slider('Ground Service (1-5)', 1, 5)

    Type_Of_Traveller = st.sidebar.selectbox('Type of Traveller',
                                             ['Business', 'Couple Leisure', 'Family Leisure', 'Solo Leisure'])
    Seat_Type = st.sidebar.selectbox('Seat Type',
                                     ['Economy Class', 'Premium Economy', 'Business Class', 'First Class'])

    traveller_encoding = {
        'Business': [1,0,0,0],
        'Couple Leisure': [0,1,0,0],
        'Family Leisure': [0,0,1,0],
        'Solo Leisure': [0,0,0,1]
    }
    traveller_encoded = traveller_encoding[Type_Of_Traveller]

    seat_encoding = {
        'Economy Class': 0,
        'Premium Economy': 1,
        'Business Class': 2,
        'First Class': 3
    }
    seat_encoded = seat_encoding[Seat_Type]

    features = np.array([[Overall_Rating, Seat_Comfort, Cabin_Staff_Service, Food_Beverages, Ground_Service,
                          traveller_encoded[0], traveller_encoded[1], traveller_encoded[2], traveller_encoded[3],
                          seat_encoded]])

    return features

st.title('AviSent')
st.write('Will the customer recommend the airline based on their experience?')

input_features = user_input_features()

input_features_scaled = input_features.copy()
input_features_scaled[:, :5] = scaler.transform(input_features[:, :5])

if st.button('Predict Recommendation'):
    prediction = model.predict(input_features_scaled)
    prediction_proba = model.predict_proba(input_features_scaled)

    if prediction[0] == 1:
        st.success(f"The customer is **LIKELY** to recommend the airline")
    else:
        st.error(f"The customer is **NOT likely** to recommend the airline")

review_text = st.text_input('Review')

if st.button('Predict sentiment'):
    review = pd.Series([review_text])
    prediction = nlp.predict(review)

    label_mapping = {0: 'not recommend', 1: 'recommend'}
    predicted_label = label_mapping[prediction[0]]

    if prediction[0] == 1:
        st.success(f"SENTIMENT: The customer is **LIKELY** to recommend the airline")
    else:
        st.error(f"SENTIMENT: The customer is **NOT likely** to recommend the airline")
