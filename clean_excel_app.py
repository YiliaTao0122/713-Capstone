import pandas as pd
import numpy as np
import streamlit as st
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Title
st.title("Comprehensive Soil Data Cleaning App")

# Step 1: File Upload
uploaded_file = st.file_uploader("Upload your soil data Excel file", type=["xlsx"])

if uploaded_file:
    # Load the Excel file
    df = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(df)

    # Step 2: Drop rows with missing critical values
    st.write("Step 2: Removing rows with missing critical values...")
    critical_columns = ['pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD']
    df_cleaned = df.dropna(subset=critical_columns, how='any')
    rows_removed = len(df) - len(df_cleaned)
    st.write(f"Rows removed due to missing critical values: {rows_removed}")

    # Step 3: Split 'Site No.1' into year, site number, and detection number
    st.write("Step 3: Splitting 'Site No.1' into 'year', 'site number', and 'detection number'...")
    if 'Site No.1' in df_cleaned.columns:
        try:
            df_cleaned[['year', 'site number', 'detection number']] = df_cleaned['Site No.1'].str.split('-', expand=True)
            st.write("Successfully split 'Site No.1' into separate columns:")
            st.dataframe(df_cleaned[['Site No.1', 'year', 'site number', 'detection number']].head())
        except Exception as e:
            st.error(f"Error splitting 'Site No.1': {e}")
    else:
        st.warning("'Site No.1' column not found. Cannot split.")

    # Step 4: Add Period Column
    st.write("Step 4: Adding Period Column...")
    conditions = [
        (df_cleaned['year'].astype(float) >= 1995) & (df_cleaned['year'].astype(float) <= 2000),
        (df_cleaned['year'].astype(float) >= 2008) & (df_cleaned['year'].astype(float) <= 2012),
        (df_cleaned['year'].astype(float) >= 2013) & (df_cleaned['year'].astype(float) <= 2017),
        (df_cleaned['year'].astype(float) >= 2018) & (df_cleaned['year'].astype(float) <= 2023)
    ]
    period_labels = ['1995-2000', '2008-2012', '2013-2017', '2018-2023']
    df_cleaned['Period'] = np.select(conditions, period_labels, default='Unknown')
    st.write("Period Column Added:")
    st.dataframe(df_cleaned.head())

    # Step 5: Replace '<' Values
    st.write("Step 5: Processing '<' values...")
    columns_with_less_than = [
        col for col in df_cleaned.columns if df_cleaned[col].astype(str).str.contains('<').any()
    ]
    for col in columns_with_less_than:
        df_cleaned[col] = df_cleaned[col].apply(
            lambda x: float(x[1:]) / 2 if isinstance(x, str) and x.startswith('<') else x
        )
    st.write(f"Processed columns with '<' values: {columns_with_less_than}")

    # Step 6: Missing Value Imputation (Optimized)
    st.write("Step 6: Filling missing values using Iterative Imputer...")

    # Impute missing values
    imputer = IterativeImputer(random_state=42, max_iter=5, tol=1e-3)
    imputed_data = imputer.fit_transform(df_cleaned)

    # Convert back to a DataFrame
    df_imputed = pd.DataFrame(imputed_data, columns=df_cleaned.columns)

    non_predictive_columns = ['Site No.1', 'year', 'site number', 'detection number', 'Period']
    df_for_imputation = df_cleaned.drop(columns=non_predictive_columns, errors='ignore')

    categorical_columns = df_for_imputation.select_dtypes(include=['object', 'category']).columns.tolist()
    with st.spinner("Performing imputation..."):
        df_imputed = perform_imputation(df_for_imputation, categorical_columns)
    st.success("Imputation complete!")
    st.write("Data after imputation:")
    st.dataframe(df_imputed)

    # Step 7: Reintegrate Data
    df_final = pd.concat([df_cleaned[non_predictive_columns].reset_index(drop=True), df_imputed], axis=1)

    # Step 8: Calculate Contamination Index (CI) and ICI
    st.write("Step 8: Calculating Contamination Index (CI) and Integrated Contamination Index (ICI)...")
    native_means = {
        "As": 6.2, "Cd": 0.375, "Cr": 28.5, "Cu": 23.0, "Ni": 17.95, "Pb": 33.0, "Zn": 94.5
    }

    for element, mean_value in native_means.items():
        if element in df_final.columns:
            df_final[f"CI_{element}"] = (df_final[element] / mean_value).round(2)

    ci_columns = [f"CI_{e}" for e in native_means.keys() if f"CI_{e}" in df_final.columns]
    if ci_columns:
        df_final['ICI'] = df_final[ci_columns].mean(axis=1).round(2)
        df_final['ICI_Class'] = df_final['ICI'].apply(
            lambda x: 'Low' if x <= 1 else 'Moderate' if x <= 3 else 'High'
        )
        st.success("ICI and contamination classification calculated successfully!")
        st.write("ICI and Contamination Classification:")
        st.dataframe(df_final[['Site No.1', 'ICI', 'ICI_Class']].head())

    # Step 9: Download Cleaned Data
    st.write("Final Cleaned Data:")
    st.dataframe(df_final)
    cleaned_file = df_final.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data",
        data=cleaned_file,
        file_name="cleaned_soil_data_with_ici.csv",
        mime="text/csv",
    )
    st.success("Data cleaning process complete! Use the button above to download the cleaned dataset.")
