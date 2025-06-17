import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import os

model = tf.keras.models.load_model("model.keras")
scaler = joblib.load("scaler.pkl")

st.title("🎓 Pass Prediction App")
st.markdown("Predict the probability of a student passing based on age and score.")

age = st.slider("Select Student Age", min_value=10, max_value=25, value=18)
score = st.slider("Select Test Score", min_value=0, max_value=100, value=75)

if st.button("predict pass probability"):
    input_feature = np.array([[age,score]])
    input_scaled = scaler.transform(input_feature)
    prediction = model.predict(input_scaled)[0][0]

    st.success(f"📈 Pass Probability: `{prediction:.2f}`")
    if prediction>0.5:
        st.balloons()
        st.markdown("### 🟢 Likely to Pass!")
    else:
        st.markdown("### 🔴 At Risk of Failing.")