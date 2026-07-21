import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit page configuration
st.set_page_config(
    page_title="Spam Email Detection",
    layout="centered"
)

# Title
st.title("Spam Email Detection")

st.write("Enter an email or SMS message to check whether it is Spam or Ham.")

# User Input
message = st.text_area("Enter your message")

# Prediction
if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Convert text into TF-IDF features
        data = vectorizer.transform([message])

        # Predict
        prediction = model.predict(data)

        # Display result
        if prediction[0] == 1:
            st.error("Spam Message")
        else:
            st.success("Ham (Not Spam)")