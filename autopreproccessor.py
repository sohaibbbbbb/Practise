import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def run_auto_eda(df):
    """Performs a quick, logical summary of the dataset."""
    print("="*50)
    print("📊 AUTOMATED EDA REPORT")
    print("="*50)
    print(f"Shape of dataset: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("--- Data Types & Missing Values ---")
    missing_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Percentage (%)': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(missing_info)
    
    print("\n--- Numerical Distribution Summary ---")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        print(df[num_cols].describe().T[['mean', 'std', 'min', '50%', 'max']])
    else:
        print("No numerical columns found.")

    print("\n--- Categorical Cardinality Summary ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        for col in cat_cols:
            print(f"{col}: {df[col].nunique()} unique values")
    else:
        print("No categorical columns found.")
    print("="*50 + "\n")

def preprocess_data(df, target_column=None):
    """
    Handles missing values, scales numerical features, 
    and encodes categorical features logically.
    """
    print("⚙️ Starting Preprocessing...")
    
    # Separate target if provided
    if target_column and target_column in df.columns:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    else:
        X = df.copy()
        y = None

    # Identify column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Define logical pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='passthrough')

    # Execute transformation
    X_processed = preprocessor.fit_transform(X)
    
    # Reconstruct DataFrame with clean feature names
    encoded_cat_cols = (
        preprocessor.named_transformers_['cat']
        .named_steps['encoder']
        .get_feature_names_out(cat_cols)
        if cat_cols else []
    )
    all_features = num_cols + list(encoded_cat_cols)
    
    X_processed_df = pd.DataFrame(X_processed, columns=all_features)
    print("✅ Preprocessing complete.")
    
    return X_processed_df, y

if __name__ == "__main__":
    # Target file path (Replace with your actual file path)
    FILE_PATH = "your_data.csv" 
    TARGET = "target_column_name" # Change or set to None if clustering/unsupervised
    
    if not os.path.exists(FILE_PATH):
        print(f"❌ Error: {FILE_PATH} not found. Please update FILE_PATH in the script.")
    else:
        # Load Data
        df = pd.read_csv(FILE_PATH)
        
        # Run Pipeline
        run_auto_eda(df)
        X_clean, y_clean = preprocess_data(df, target_column=TARGET)
        
        # Output verification
        print(f"\nFinal Feature Matrix Shape: {X_clean.shape}")