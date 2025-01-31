import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Loads the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset as a DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The input file was not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError("The dataset is empty. Please check the input file.")
        print("Dataset loaded successfully. Columns:", df.columns)
        return df
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the data: {e}")

def clean_data(df):
    """
    Cleans the dataset by removing null values and duplicates.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df = df.dropna()
    df = df.drop_duplicates()

    return df

def preprocess_data(df):
    """
    Preprocesses the dataset by encoding sentiment and splitting into train/test sets.

    Args:
        df (pd.DataFrame): Input dataset.

    Returns:
        tuple: Training and test datasets (X_train, X_test, y_train, y_test).
    """

    # Ensure 'sentiment' column exists
    if 'sentiment' not in df.columns:
        raise ValueError("The dataset must contain a 'sentiment' column.")

    # Create 'label' column if it doesn't exist
    if 'label' not in df.columns:
        print("'label' column not found. Creating 'label' column based on 'sentiment'.")
        df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    if 'review' not in df.columns or 'label' not in df.columns:
        raise ValueError("The dataset must contain both 'review' and 'label' columns.")

    # Debugging output for the 'label' column
    print("Columns after processing:", df.columns)
    print("Sample of 'label' column:", df['label'].head())

    X = df['review']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def save_cleaned_data(df, output_path):
    """
    Saves the cleaned dataset to a CSV file with only 'review' and 'label' columns.

    Args:
        df (pd.DataFrame): Cleaned dataset.
        output_path (str): Path to save the cleaned dataset.
    """
    output_dir = os.path.dirname(output_path)
    print(f"Checking if directory {output_dir} exists.")
    if not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist. Creating it now.")
        os.makedirs(output_dir, exist_ok=True)

    try:
        # Select only required columns
        if 'review' not in df.columns or 'label' not in df.columns:
            raise ValueError("The cleaned dataset does not contain the required columns 'review' and 'label'.")

        df_to_save = df[['review', 'label']]

        if df_to_save.empty:
            raise ValueError("The DataFrame is empty. Cannot save empty data.")
        df_to_save.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
    except KeyError as e:
        print(f"Error: Required columns missing from dataset: {e}")
        raise
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

if __name__ == "__main__":
    # Define file paths
    input_file = "data/raw/IMDB Dataset.csv"
    output_file = "data/cleaned_data/cleaned_data.csv"

    try:
        # Load the dataset
        data = load_data(input_file)

        # Clean the dataset
        data_cleaned = clean_data(data)

        # Preprocess the dataset for training
        X_train, X_test, y_train, y_test = preprocess_data(data_cleaned)

        # Save cleaned data
        save_cleaned_data(data_cleaned, output_file)

        print("Data preprocessing completed. Cleaned data saved to:", output_file)
    except Exception as e:
        print(f"An error occurred: {e}")