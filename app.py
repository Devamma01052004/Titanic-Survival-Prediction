import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter the passenger details to predict survival:")

# Collect user input
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.2)

# Encode sex (IMPORTANT)
sex_encoded = 0 if sex == "male" else 1

# Predict button
if st.button("Predict"):
    # Create DataFrame exactly as model expects
    input_data = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display numeric output
    st.write("### 🔢 Model Prediction Output")
    st.write(f"**{prediction}**  (0 = Did Not Survive, 1 = Survived)")

    # Display message
    if prediction == 1:
        st.success("🎉 The passenger **survived!**")
    else:
        st.error("💀 The passenger **did not survive.**")
