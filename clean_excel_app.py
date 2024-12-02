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

    # Step 3: Extract and Process Sample Count
    st.write("Step 3: Extracting Sample Count...")
    if 'Site No.1' in df_cleaned.columns:
        df_cleaned['Sample Count'] = df_cleaned['Site No.1'].str.extract(r'-(\d{2})$').astype(float)
        st.write("Sample Count Extracted:")
        st.dataframe(df_cleaned[['Site No.1', 'Sample Count']].head())

    # Step 4: Add Period Column
    st.write("Step 4: Adding Period Column...")
    conditions = [
        (df_cleaned['Year'] >= 1995) & (df_cleaned['Year'] <= 2000),
        (df_cleaned['Year'] >= 2008) & (df_cleaned['Year'] <= 2012),
        (df_cleaned['Year'] >= 2013) & (df_cleaned['Year'] <= 2017),
        (df_cleaned['Year'] >= 2018) & (df_cleaned['Year'] <= 2023)
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

    # Step 6: Missing Value Imputation
    st.write("Step 6: Filling missing values using Iterative Imputer...")
    non_predictive_columns = ['Site No.1', 'Site Num', 'Year', 'Sample Count', 'Period']
    df_for_imputation = df_cleaned.drop(columns=non_predictive_columns, errors='ignore')

    # Ensure only numeric columns are passed for imputation
    numeric_columns = df_for_imputation.select_dtypes(include=[np.number]).columns
    df_for_imputation = df_for_imputation[numeric_columns]

    # Drop rows with all NaNs
    df_for_imputation = df_for_imputation.dropna(how='all', axis=0)

    # Display data before imputation
    st.write("Data before imputation (formatted to 2 decimal places):")
    st.dataframe(df_for_imputation.style.format("{:.2f}"))

    # Perform imputation
    try:
        imputer = IterativeImputer(random_state=0, max_iter=50, tol=1e-4)
        imputed_array = imputer.fit_transform(df_for_imputation)
        df_imputed = pd.DataFrame(imputed_array, columns=df_for_imputation.columns)

        # Round imputed values to 2 decimal places
        df_imputed = df_imputed.round(2)

        # Show data after imputation
        st.write("Data after imputation (formatted to 2 decimal places):")
        st.dataframe(df_imputed.style.format("{:.2f}"))

        # Update the original DataFrame with imputed values
        df_cleaned.update(df_imputed)
        st.success("Missing values filled successfully!")
    except ValueError as e:
        st.error(f"Imputation failed: {e}")
        st.stop()

    # Step 7: Calculate Contamination Index (CI) and ICI
    st.write("Step 7: Calculating Contamination Index (CI) and Integrated Contamination Index (ICI)...")

    # Define native means for elements
    native_means = {
        "As": 6.2, "Cd": 0.375, "Cr": 28.5, "Cu": 23.0, "Ni": 17.95, "Pb": 33.0, "Zn": 94.5
    }

    # Ensure required columns are present
    required_elements = list(native_means.keys())
    missing_columns = [col for col in required_elements if col not in df_cleaned.columns]

    if missing_columns:
        st.warning(f"Missing columns for ICI calculation: {', '.join(missing_columns)}")
    else:
        # Calculate CI for each element
        for element, mean_value in native_means.items():
            if element in df_cleaned.columns:
                df_cleaned[f"CI_{element}"] = (df_cleaned[element] / mean_value).round(2)

        # Calculate ICI
        if all(f"CI_{e}" in df_cleaned.columns for e in required_elements):
            df_cleaned['ICI'] = df_cleaned[[f"CI_{e}" for e in required_elements]].mean(axis=1).round(2)

            # Classify ICI
            def classify_ici(ici):
                if ici <= 1:
                    return "Low Contamination"
                elif 1 < ici <= 3:
                    return "Moderate Contamination"
                else:
                    return "High Contamination"

            df_cleaned['ICI_Class'] = df_cleaned['ICI'].apply(classify_ici)

            # Display ICI results
            st.success("ICI and contamination classification calculated successfully!")
            st.write("ICI and Contamination Classification:")
            st.dataframe(df_cleaned[['Site No.1', 'ICI', 'ICI_Class']].head())
        else:
            st.error("ICI calculation failed due to missing CI columns.")

    # Step 8: Display Final Cleaned Data
    st.write("Final Cleaned Data:")
    st.dataframe(df_cleaned)

    # Step 9: Download Cleaned Data
    cleaned_file = df_cleaned.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data",
        data=cleaned_file,
        file_name="cleaned_soil_data_with_ici.csv",
        mime="text/csv",
    )
    st.success("Data cleaning process complete! Use the button above to download the cleaned dataset.")
