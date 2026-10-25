import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("🧳 Wellness Tourism Package Predictor")
st.write("Enter customer details to predict whether they will purchase the "
         "**Wellness Tourism Package**.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run the GitHub Actions pipeline to train and commit it.")
    st.stop()

model = load_model()

# ---------------- Inputs ----------------
st.subheader("Customer details")
c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    income = st.number_input("Monthly Income", 1000, 100000, 20000, step=500)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    passport = st.selectbox("Holds Passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns Car?", ["No", "Yes"])
with c2:
    persons = st.number_input("Number of People Visiting", 1, 10, 3)
    children = st.number_input("Children Visiting (under 5)", 0, 5, 0)
    trips = st.number_input("Trips per Year", 1, 25, 3)
    star = st.selectbox("Preferred Property Star", [3, 4, 5])

st.subheader("Interaction details")
c3, c4 = st.columns(2)
with c3:
    contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Duration of Pitch (minutes)", 1, 140, 15)
with c4:
    followups = st.number_input("Number of Follow-ups", 1, 10, 3)
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# Collect inputs into a dataframe (same columns as training data)
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons,
    "NumberOfFollowups": followups,
    "ProductPitched": product,
    "PreferredPropertyStar": star,
    "MaritalStatus": marital,
    "NumberOfTrips": trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": satisfaction,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": children,
    "Designation": designation,
    "MonthlyIncome": income,
}])

with st.expander("Input data sent to the model"):
    st.dataframe(input_df)

# ---------------- Prediction ----------------
if st.button("Predict"):
    proba = model.predict_proba(input_df)[0, 1]
    pred = int(proba >= 0.5)
    st.metric("Purchase probability", f"{proba:.1%}")
    if pred == 1:
        st.success("✅ Likely to purchase the Wellness Tourism Package. Worth contacting.")
    else:
        st.warning("❌ Unlikely to purchase the Wellness Tourism Package.")
