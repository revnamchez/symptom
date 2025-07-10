import streamlit as st
import joblib
import pandas as pd
import random


# Load the trained model and vectorizer
try:
    model = joblib.load("log_symptom.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found.")
    st.stop()

# Streamlit app
st.set_page_config(page_title="Sickness-Prediction System")
st.title("Sickness-Prediction System")

# Tab selection
tab1, tab2 = st.tabs(["Sickness Prediction", "Get Advice"])



with tab1:
    st.markdown("Describe your symptoms in detail:")
    # Example symptoms for user guidance
    st.info("**Example:** \"I have a red, itchy skin rash on my arms and legs, covered in dry, scaly patches.\"")
    symptoms_input = st.text_area(
        "Type below:",
        placeholder="e.g., 'I have a persistent cough, fever, and body aches.'",
        height=150
    )

    if st.button("Predict Sickness"):
        if symptoms_input.strip() == "":
            st.warning("Please enter some symptoms to get a prediction.")
        else:
            with st.spinner("Analyzing your symptoms..."):
                try:
                    symptoms_tfidf = vectorizer.transform([symptoms_input])
                    prediction = model.predict(symptoms_tfidf)[0]
                    probabilities = model.predict_proba(symptoms_tfidf)[0]
                    sickness_probs = list(zip(model.classes_, probabilities))
                    sickness_probs.sort(key=lambda x: x[1], reverse=True)
                   
                    st.success(f"### Primary Prediction: {prediction}")
                    st.write(f"Confidence: *{(max(probabilities) * 100):.1f}%*")
                    st.markdown("### Top 3 Predictions:")
                    for i, (sickness, prob) in enumerate(sickness_probs[:3]):
                        st.write(f"*{i+1}. {sickness}*: {(prob * 100):.1f}%")
                        
                    st.markdown(
                        """ 
                        --- 
                        *Important Disclaimer:* 
                        This is an AI-based prediction tool for educational purposes only. 
                        Please consult with a qualified healthcare professional for proper medical diagnosis and treatment. 
                        """
                    )
                            

                except Exception as e:
                    st.error(f"An error occured during prediction: {e}")

                                                                  




with tab2:
    
        responses = {
            "major": ["Has the symptom persisted for over 24 hours?"],
            "minor": ["Has the symptom persisted for over 24 hours?"],
            "beta": ["You may need some rest while you monitor the situation. ok?"],
            "wos": ["you really need urgent medical attention. See your doctor. ok?"],
            "affirm": ["up till now, do you feel 'better' or 'worse'"],
            "naffirm": ["please avoid self medication. See your doctor if it persists after 24 hours. ok?"],
            "often": ["How often do you fall sick? ('rarely', 'occasionally', 'frequently')"],
            "rare": ["You seems to have a strong immune system. Wait and observe, but seek medical attention if symptoms persists after 48 hours. Thanks."],
            "ocassion": ["You seems to have an average immune system. Avoid stress, and seek medical attention if symptoms persists after 24 hours. Thanks."],
            "freq": ["Your immune system is probably weak. Please see your doctor at once. Thanks."],
            "semi": ["You are welcome. Take care of yourself and goodbye."],
            "end": ["🤖👋👋👋"]
        }
    
        #Keywords
        keywords = {
            "major": ["major"],
            "minor": ["minor"],
            "beta": ["better"],
            "wos": ["worse"],
            "affirm": ["yes"],
            "naffirm": ["no"],
            "often": ["ok", 'okay'],
            "rare": ["rarely"],
            "ocassion": ["occasionally"],
            "freq": ["frequently"],
            "semi": ["thanks", "thank you"],
            "end": ["goodbye", "bye"]
        }


def get_response(user_input):
    user_input = user_input.lower()
        
    for intent, keys in keywords.items():
        for key in keys:
            if key in user_input:
                if intent == "end":
                    return responses[intent][0] + " " + ""
                else:
                    return random.choice(responses[intent])
        
    return "Please type in lower case and follow my simple response guide."
    


# Frontend
def main():
        st.subheader("Get Advice - Mini ChatBot 🤖")
        st.write("Rate your Symptom (major or minor?)")
    
        user_input = st.text_input("type in small letters:")
        
        if user_input:
            response = get_response(user_input)
            st.write("ChatBot:", response)
        

# Run the app
if __name__ == "__main__":
        main()


st.markdown(
            """ 
            --- 
            *Important Disclaimer:* 
            This is an AI-based prediction tool for educational purposes only. 
            Please consult with a qualified healthcare professional for proper medical diagnosis and treatment. 
            """
    )


