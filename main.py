"""
Main script to orchestrate the data processing, model training, and evaluation pipeline
for the energy consumption prediction project.

This script:
1. Loads and preprocesses the data.
2. Applies optional data augmentation.
3. Trains the BERT-based text classification model.
4. Evaluates the model and saves the results.
5. Saves the best-performing model.

Author: [Your Name]
Date: [Current Date]
"""

import os
import pandas as pd
import tensorflow as tf
from src.preprocess_data import preprocess_data
from src.train_model import train_bert_model
from src import config
from src.augment_data import augment_texts
from sklearn.model_selection import train_test_split

# Suppress TensorFlow warnings using the level from config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL
tf.get_logger().setLevel('ERROR')

# File paths
RAW_DATA_PATH = "data/raw/IMDB Dataset.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
OUTPUT_MODEL_PATH = "models/bert_text_classifier"
RESULTS_CSV_PATH = "results/training_results.csv"

# Ensure output directories exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)

def main():
    """
    Main function to run the entire machine learning pipeline.
    """

    # Step 1: Data Preprocessing
    print("\nStarting data preprocessing...")
    data = preprocess_data(RAW_DATA_PATH, PROCESSED_DATA_PATH)

    # Step 2: Verify dataset structure
    required_columns = ["review", "label"]
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"The dataset must contain the following columns: {required_columns}")

    # Step 3: Reduce dataset size if TPU is unavailable
    if not any(device.device_type == "TPU" for device in os.environ.get('TF_CPP_MIN_LOG_LEVEL', [])):
        print("TPU not available. Reducing dataset size for CPU/GPU usage...")
        reduced_size = int(len(data) * (config.REDUCTION_PERCENTAGE / 100))
        data = data.sample(n=reduced_size, random_state=42).reset_index(drop=True)
        print(f"Dataset reduced to {config.REDUCTION_PERCENTAGE}% of its original size: {len(data)} rows remaining.")

    # Step 4: Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        data["review"].tolist(),  # Convert to list to avoid tokenizer issues
        data["label"].tolist(),
        test_size=0.2,
        random_state=42
    )

    # Step 5: Apply data augmentation if enabled in config
    if config.USE_AUGMENTATION:
        print("Applying data augmentation...")
        X_train = augment_texts(X_train, aug_factor=config.AUGMENT_FACTOR)

    # Step 6: Define hyperparameter search space
    learning_rates = [1e-5, 2e-5, 5e-5]
    batch_sizes = [16, 32, 64]

    # Step 7: Train and evaluate models
    best_model = None
    best_accuracy = 0.0
    best_hyperparams = {}

    results = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"\nTraining with learning rate: {lr} and batch size: {batch_size}")
            val_acc, val_loss, model = train_bert_model(
                X_train, y_train, X_val, y_val,
                learning_rate=lr, batch_size=batch_size,
                weight_decay=0.01, clip_norm=1.0
            )

            # Store results
            results.append({
                "learning_rate": lr,
                "batch_size": batch_size,
                "val_accuracy": val_acc,
                "val_loss": val_loss.numpy()
            })

            # Track the best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model = model
                best_hyperparams = {"learning_rate": lr, "batch_size": batch_size}

    # Step 8: Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nTraining results saved to {RESULTS_CSV_PATH}")

    # Step 9: Save the best model
    if best_model:
        best_model.save_pretrained(OUTPUT_MODEL_PATH)
        print(f"\nBest model saved with LR: {best_hyperparams['learning_rate']} and Batch Size: {best_hyperparams['batch_size']}")

if __name__ == "__main__":
    main()