import streamlit as st
import tensorflow as tf
import numpy as np
import random

# Load the trained model
try:
    model = tf.keras.models.load_model("fake_profile_model.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Fake Profile Detection")
st.write("Enter the details below to check if a profile is fake.")

# Define feature names
feature_names = [
    "Profile Age (Days)", "Number of Followers", "Number of Following", "Posts Count", "Bio Length",
    "Has Profile Picture (0/1)", "Engagement Rate", "Average Likes Per Post", "Average Comments Per Post",
    "Account Verification Status (0/1)", "Suspicious Activity Score"
]

# User inputs
profile_link = st.text_input("Enter Instagram Profile Link (Optional)")
fetched_data = None

if profile_link:
    st.write("Fetching profile data...")

    # Simulate fetching data - Randomly decide if the profile is real or fake
    is_fake = random.choice([True, False])

    fetched_data = {
        "Profile Age (Days)": random.randint(0, 30) if is_fake else random.randint(365, 2000),
        "Number of Followers": random.randint(10, 500) if is_fake else random.randint(1000, 100000),
        "Number of Following": random.randint(500, 5000) if is_fake else random.randint(100, 1000),
        "Posts Count": random.randint(0, 5) if is_fake else random.randint(50, 5000),
        "Bio Length": random.randint(0, 20) if is_fake else random.randint(50, 200),
        "Has Profile Picture (0/1)": 0 if is_fake else 1,
        "Engagement Rate": round(random.uniform(0.1, 0.5) if is_fake else random.uniform(1.5, 5.0), 2),
        "Average Likes Per Post": random.randint(1, 10) if is_fake else random.randint(100, 10000),
        "Average Comments Per Post": random.randint(0, 2) if is_fake else random.randint(10, 500),
        "Account Verification Status (0/1)": 0 if is_fake else random.choice([0, 1]),
        "Suspicious Activity Score": round(random.uniform(0.7, 1.0) if is_fake else random.uniform(0.0, 0.3), 2)
    }

# Create input fields
features = []
for name in feature_names:
    default_value = fetched_data[name] if fetched_data else ""
    user_input = st.text_input(name, value=str(default_value))

    try:
        features.append(float(user_input))
    except ValueError:
        st.error(f"Invalid input for {name}. Please enter a valid number.")
        st.stop()

# Convert input to numpy array
input_data = np.array(features).reshape(1, -1)

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_data)[0]  # Get the two-class output
    predicted_class = int(np.argmax(prediction))  # Get the class index (0 or 1)
    confidence = float(np.max(prediction))  # Confidence score

    st.write(f"Prediction Confidence: {confidence:.4f}")

    if confidence > 0.75:
        if predicted_class == 1:
            st.error("⚠️ This is likely a FAKE profile!")
        else:
            st.success("✅ This appears to be a REAL profile.")
    else:
        st.warning("⚠️ Uncertain prediction. Confidence is low!")
