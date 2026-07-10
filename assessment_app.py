
import streamlit as st
import pickle
import pandas as pd

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load label encoders
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Page Configuration
st.set_page_config(
    page_title="Student Career Role Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Career Role Prediction System")

st.write("Select your proficiency level in each subject to predict the most suitable career role.")

# Available Levels
levels = ["Beginner", "Intermediate", "Professional"]

# User Inputs
database = st.selectbox("Database Fundamentals", levels)

computer_architecture = st.selectbox("Computer Architecture", levels)

distributed = st.selectbox("Distributed Computing Systems", levels)

cyber = st.selectbox("Cyber Security", levels)

networking = st.selectbox("Networking", levels)

software_dev = st.selectbox("Software Development", levels)

programming = st.selectbox("Programming Skills", levels)

project_management = st.selectbox("Project Management", levels)

forensics = st.selectbox("Computer Forensics Fundamentals", levels)

technical = st.selectbox("Technical Communication", levels)

aiml = st.selectbox("AI ML", levels)

software_engineering = st.selectbox("Software Engineering", levels)

business = st.selectbox("Business Analysis", levels)

communication = st.selectbox("Communication Skills", levels)

datascience = st.selectbox("Data Science", levels)

troubleshooting = st.selectbox("Troubleshooting Skills", levels)

graphics = st.selectbox("Graphics Designing", levels)


# Prediction
if st.button("Predict Career Role"):

    student = pd.DataFrame({
        "Database Fundamentals":[database],
        "Computer Architecture":[computer_architecture],
        "Distributed Computing Systems":[distributed],
        "Cyber Security":[cyber],
        "Networking":[networking],
        "Software Development":[software_dev],
        "Programming Skills":[programming],
        "Project Management":[project_management],
        "Computer Forensics Fundamentals":[forensics],
        "Technical Communication":[technical],
        "AI ML":[aiml],
        "Software Engineering":[software_engineering],
        "Business Analysis":[business],
        "Communication skills":[communication],
        "Data Science":[datascience],
        "Troubleshooting skills":[troubleshooting],
        "Graphics Designing":[graphics]
    })

    # Encode Inputs
    for col in student.columns:
        student[col] = label_encoders[col].transform(student[col])

    # Prediction
    prediction = model.predict(student)

    role = label_encoders["Role"].inverse_transform(prediction)

    st.success(f"🎯 Predicted Career Role: {role[0]}")

    st.balloons()
