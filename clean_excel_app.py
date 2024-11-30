import pandas as pd
import streamlit as st
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer

# Streamlit App Title
st.title("Soil Data Cleaner")

# File Upload
uploaded_file = st.file_uploader("Upload your soil data Excel file", type=["xlsx"])

if uploaded_file:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)

    # Step 1: Preprocess '<' Values
    st.write("Processing '<' values...")
    columns_with_less_than = [
        col for col in df.columns if df[col].astype(str).str.contains('<').any()
    ]
    for col in columns_with_less_than:
        df[col] = df[col].apply(
            lambda x: float(x[1:]) / 2 if isinstance(x, str) and x.startswith('<') else x
        )
    st.success("Finished processing '<' values!")

    # Step 2: Handle Missing Values
    st.write("Filling missing values...")
    columns_to_impute = ['pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD', 'MP-5', 'MP-10', 
                         'As', 'Cd', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']
    try:
        # Convert target columns to numeric
        df[columns_to_impute] = df[columns_to_impute].apply(pd.to_numeric, errors='coerce')
        
        # Use IterativeImputer to fill missing values
        imputer = IterativeImputer(random_state=0, max_iter=50, tol=1e-4)
        df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
        st.success("Missing values filled!")
    except Exception as e:
        st.error(f"Error while filling missing values: {e}")

    # Step 3: Display and Download Cleaned Data
    st.write("Cleaned Data:")
    st.dataframe(df)

    # Save the cleaned data as a CSV file
    cleaned_file = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned File",
        data=cleaned_file,
        file_name="cleaned_soil_data.csv",
        mime="text/csv",
    )
    st.success("Data cleaning complete! Use the button above to download the cleaned file.")
