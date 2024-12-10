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

    # Step 6: Missing Value Imputation (Numeric and Categorical)
    st.write("Step 6: Filling missing values using Iterative Imputer...")
    non_predictive_columns = ['Site No.1', 'year', 'site number', 'detection number', 'Period']
    df_for_imputation = df_cleaned.drop(columns=non_predictive_columns, errors='ignore')

    # Separate categorical and numeric columns
    categorical_columns = df_for_imputation.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = df_for_imputation.select_dtypes(include=[np.number]).columns.tolist()

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_for_imputation, columns=categorical_columns, drop_first=False)

    try:
        # Apply IterativeImputer
        imputer = IterativeImputer(random_state=42, max_iter=10)
        imputed_data = imputer.fit_transform(df_encoded)

        # Convert the imputed data back to a DataFrame
        df_imputed = pd.DataFrame(imputed_data, columns=df_encoded.columns)

        # Map one-hot-encoded columns back to original categorical columns
        for col in categorical_columns:
            encoded_columns = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
            df_imputed[col] = df_imputed[encoded_columns].idxmax(axis=1).str[len(col) + 1:]
            df_imputed = df_imputed.drop(columns=encoded_columns)

        # Reattach non-predictive columns to the imputed dataset
        df_final = pd.concat([df_cleaned[non_predictive_columns].reset_index(drop=True), df_imputed], axis=1)

        st.success("Missing values filled successfully!")
        st.write("Data after imputation:")
        st.dataframe(df_final)

    except ValueError as e:
        st.error(f"Imputation failed: {e}")
        st.stop()

    # Step 7: Calculate Contamination Index (CI) and ICI
    st.write("Step 7: Calculating Contamination Index (CI) and Integrated Contamination Index (ICI)...")
    native_means = {
        "As": 6.2, "Cd": 0.375, "Cr": 28.5, "Cu": 23.0, "Ni": 17.95, "Pb": 33.0, "Zn": 94.5
    }
    required_elements = list(native_means.keys())
    missing_columns = [col for col in required_elements if col not in df_final.columns]

    if missing_columns:
        st.warning(f"Missing columns for ICI calculation: {', '.join(missing_columns)}")
    else:
        for element, mean_value in native_means.items():
            if element in df_final.columns:
                df_final[f"CI_{element}"] = (df_final[element] / mean_value).round(2)

        if all(f"CI_{e}" in df_final.columns for e in required_elements):
            df_final['ICI'] = df_final[[f"CI_{e}" for e in required_elements]].mean(axis=1).round(2)

            def classify_ici(ici):
                if ici <= 1:
                    return "Low"
                elif 1 < ici <= 3:
                    return "Moderate"
                else:
                    return "High"

            df_final['ICI_Class'] = df_final['ICI'].apply(classify_ici)

            st.success("ICI and contamination classification calculated successfully!")
            st.write("ICI and Contamination Classification:")
            st.dataframe(df_final[['Site No.1', 'ICI', 'ICI_Class']].head())
        else:
            st.error("ICI calculation failed due to missing CI columns.")

    # Step 8: Display Final Cleaned Data
    st.write("Final Cleaned Data:")
    st.dataframe(df_final)

    # Step 9: Download Cleaned Data
    cleaned_file = df_final.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Data",
        data=cleaned_file,
        file_name="cleaned_soil_data_with_ici.csv",
        mime="text/csv",
    )
    st.success("Data cleaning process complete! Use the button above to download the cleaned dataset.")
