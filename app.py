import streamlit as st
import nltk
from transformers import pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings

import tensorflow as tf

# Suppress warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# Load the chatbot model safely
try:
    chatbot = pipeline("text-generation", model="distilgpt2")
except Exception as e:
    st.error(f"Error loading chatbot model: {e}")
    chatbot = None  # Avoid crashing the app

# Define the chatbot function
def healthcare_chatbot(user_input):
    if "symptom" in user_input.lower():
        return "Please consult a doctor for accurate advice."
    elif "appointment" in user_input.lower():
        return "Would you like to schedule an appointment with the doctor?"
    elif "medication" in user_input.lower():
        return "It's important to take prescribed medicines regularly. If you have concerns, consult your doctor."
    
    if chatbot:
        try:
            response = chatbot(user_input, max_length=500, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
            return f"Error generating response: {e}"
    else:
        return "Chatbot is not available at the moment."

# Define the main function
def main():
    st.title("Healthcare Assistant Chatbot 🤖💊")

    # Use session state to store conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Input field for user input
    user_input = st.text_area("How can I assist you today?", height=100)

    # Button to submit the input
    if st.button("Submit"):
        if user_input.strip():  # Ensure input is not empty
            # Append user input
            st.session_state.messages.append(("User", user_input))
            
            # Show spinner while processing
            with st.spinner("Processing your query, please wait..."):
                response = healthcare_chatbot(user_input)  # Get chatbot response
                st.session_state.messages.append(("Healthcare Assistant", response))

        else:
            st.warning("Please enter a message to get a response.")

    # Display conversation history
    for role, message in st.session_state.messages:
        st.write(f"**{role}:** {message}")

# Run the app
if __name__ == "__main__":
    main()
