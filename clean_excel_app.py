import pandas as pd
import streamlit as st
from missingpy import MissForest

st.title("Soil Data Cleaner")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your soil data Excel file", type=["xlsx"])

if uploaded_file:
    # Load the Excel file
    df = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)

    # Step 2: Handle Missing Values
    st.write("Filling missing values...")
    columns_to_impute = ['pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD', 'MP-5', 'MP-10', 'As', 'Cd', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']
    try:
        imputer = MissForest()
        df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
        st.success("Missing values filled!")
    except Exception as e:
        st.error(f"Error filling missing values: {e}")

    # Step 3: Replace Values Below Detection Limits
    st.write("Adjusting values below detection limits...")
    columns_with_less_than = [col for col in df.columns if df[col].astype(str).str.contains('<').any()]
    for col in columns_with_less_than:
        df[col] = df[col].apply(lambda x: float(x[1:]) / 2 if isinstance(x, str) and x.startswith('<') else x)
    st.success("Values below detection limits adjusted!")

    # Step 4: Save Cleaned Data
    cleaned_file = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cleaned Data", cleaned_file, "cleaned_soil_data.csv", "text/csv")
    st.success("Data cleaning complete! Download your cleaned file using the button above.")
