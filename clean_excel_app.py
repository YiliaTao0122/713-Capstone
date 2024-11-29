import pandas as pd
import streamlit as st

st.title("Excel File Cleaner")

uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("Original Data:")
    st.dataframe(df.head())

    # Cleaning logic
    cleaned_df = df.drop_duplicates().dropna()
    st.write("Cleaned Data:")
    st.dataframe(cleaned_df)

    # Download the cleaned file
    cleaned_file = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Cleaned File", cleaned_file, "cleaned_data.csv", "text/csv")
