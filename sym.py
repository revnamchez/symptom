import streamlit as st
import joblib
import pandas as pd

# Load the trained model and vectorizer
# Ensure these files are in the same directory as your Streamlit app or provide full paths
try:
    model = joblib.load("log_symptom.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Please ensure 'log_symptom.pkl' and 'vectorizer.pkl' are in the correct directory.")
    st.stop()

st.set_page_config(page_title="Sickness Prediction System")

st.title("Sickness Prediction System")
st.markdown("Enter your symptoms below to get a potential sickness diagnosis.")

# Example symptoms for user guidance
st.info("**Example Symptoms:** \"I have been experiencing a skin rash on my arms and legs. It is red, itchy, and covered in dry, scaly patches. There is also joint pain in my fingers and wrists.\"")

# Text area for symptom input
symptoms_input = st.text_area(
    "Describe your symptoms in detail:",
    placeholder="e.g., 'I have a persistent cough, fever, and body aches.'",
    height=150
)

if st.button("Predict Sickness"):
    if symptoms_input.strip() == "":
        st.warning("Please enter some symptoms to get a prediction.")
    else:
        with st.spinner("Analyzing your symptoms..."):
            try:
                # Transform the symptoms using the loaded vectorizer
                symptoms_tfidf = vectorizer.transform([symptoms_input])

                # Make prediction
                prediction = model.predict(symptoms_tfidf)[0]

                # Get prediction probabilities for all diseases
                probabilities = model.predict_proba(symptoms_tfidf)[0]
                sickness_probs = list(zip(model.classes_, probabilities))
                sickness_probs.sort(key=lambda x: x[1], reverse=True)

                # Display primary prediction
                st.success(f"### Primary Prediction: {prediction}")
                st.write(f"Confidence: **{(max(probabilities) * 100):.1f}%**")

                # Display top 3 predictions
                st.markdown("### Top 3 Predictions:")
                for i, (sickness, prob) in enumerate(sickness_probs[:3]):
                    st.write(f"**{i+1}. {sickness}**: {(prob * 100):.1f}%")

                st.markdown(
                    """
                    --- 
                    **Important Disclaimer:** This is an AI-based prediction tool for educational purposes only. 
                    Please consult with a qualified healthcare professional for proper medical diagnosis and treatment.
                    """
                )

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")