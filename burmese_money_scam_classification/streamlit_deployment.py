import nltk
 
try:
    nltk.download('stopwords')
except Exception as e:
    print("Error downloading stopwords:", e)
 
import joblib
import streamlit as st
import pandas as pd
import preprocess_text 
import random

# Load Model
model = joblib.load("burmese_money_scam_classification/models/scam_detector_tfidf_word.pkl")

fraudulent_messages = [
    "⚠️ Warning! This is a well-known scam tactic. Do not engage.",
    "🚨 This message is a scam! Avoid responding or clicking any links.",
    "🔴 High-risk fraud detected! Report this message and warn others.",
    "❗ Do NOT provide personal or financial information. This is a scam.",
    "⚠️ This message is part of a fraudulent scheme. Block the sender immediately.",
    "🚫 Avoid interacting! Scammers often impersonate trusted sources.",
    "🔴 Scam alert! If something sounds too good to be true, it probably is."
]

potential_fraudulent_messages = [
    "🤔 This message seems suspicious. Verify with official sources before acting.",
    "⚠️ Be cautious! Scammers use urgency to pressure victims into quick decisions.",
    "🔍 Double-check with an official source. If unsure, do NOT engage.",
    "⚠️ This message has red flags. Don't share any personal details.",
    "🟠 This could be a scam. Cross-check before taking any action.",
    "🛑 Stop and think! If they ask for money or personal info, it's likely a scam.",
    "🔍 Verify the sender. Scammers often impersonate trusted organizations."
]

safe_messages = [
    "✅ No signs of fraud detected. However, always stay alert online.",
    "🔎 This message appears safe, but always verify important information.",
    "🟢 Looks safe! If you're ever unsure, check official sources.",
    "✅ No fraud detected. However, scams evolve—always stay cautious!",
    "🟢 The message seems legitimate, but it's always good to stay alert.",
    "🔍 No immediate fraud risk, but online safety is always important.",
    "✅ This message is classified as safe, but be mindful of phishing attempts."
]


# Dynamic Suggestion Generator
def generate_suggestion(prediction, text):

    # Fraudulent Scenarios
    if prediction == 2:
        return random.choice(fraudulent_messages)

    # Potential Fraudulent Scenarios
    elif prediction == 0:
       return random.choice(potential_fraudulent_messages)

    # Safe Scenario
    else:
        return random.choice(safe_messages)

# Function to classify message and generate output
def classify_message(text):
    # Preprocess the input text
    processed_text, emoji_count, hashtag_count, punctuation_counts = preprocess_text.preprocess_text(text)  
  
    # Convert to DataFrame to match training format
    input_df = pd.DataFrame({
        "processed_text": [processed_text],
        "emoji_count": [emoji_count],
        "hashtag_count": [hashtag_count],
        "punctuation_counts": [punctuation_counts]  
    })

    # Model Prediction
    prediction = model.predict(input_df)[0]  
    probabilities = model.predict_proba(input_df)[0]  # Get class probabilities

    # Assign Risk Level
    risk_levels = {
        2: "🟥 High Risk Scam",
        0: "🟧 Potential Scam",
        1: "🟩 Safe, Non Scam"
    }
    risk_level = risk_levels.get(prediction, "Unknown")

    # Confidence Score
    confidence = round(probabilities[prediction] * 100, 2)  # Convert to percentage

    # Generate Dynamic Suggestion
    suggestion = generate_suggestion(prediction, text)

    return prediction, risk_level,confidence, suggestion

# Streamlit UI
st.title("🛡️ FraudAlert : Burmese Money Scam Detector")

# Run in Streamlit
user_input = st.text_area("Enter a message to check for scams:")

if st.button("Check Scam"):
    if user_input.strip():
        prediction, danger_level,confidence, suggestion = classify_message(user_input)
        #st.write(f"**Predicted Label:** {prediction}")
        st.write(f"**Danger Level:** {danger_level}")
        st.write(f"**Prediction Confidence:** {confidence}%")  # Show confidence
        st.write(f"**Suggestion:** {suggestion}")
    else:
        st.warning("Please enter a message.")

# Run in CLI (Command Line)
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        prediction, danger_level,confidence, suggestion = classify_message(text)
        print(f"\nDanger Level: {danger_level}")
        print(f"Prediction Confidence: {prediction}")
        print(f"Suggestion: {suggestion}\n")
    else:
        print("Run the script with a message as an argument or use Streamlit UI.")