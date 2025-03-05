import tensorflow as tf
import os
import pandas as pd
import csv
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from sklearn.utils import shuffle
from src import config

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL

# Paths
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
OUTPUT_MODEL_PATH = "models/bert_text_classifier"
RESULTS_CSV_PATH = "results/training_results.csv"

# Ensure output directories exist
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)

# Load processed data
data = pd.read_csv(PROCESSED_DATA_PATH)

# Verify dataset structure
required_columns = ["review", "label"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The dataset must contain the following columns: {required_columns}")

# Subsampling function
def reduce_dataset(data, percentage=config.REDUCTION_PERCENTAGE):
    """Reduce the dataset size to a specified percentage."""
    reduced_size = int(len(data) * (percentage / 100))
    data = shuffle(data, random_state=42)  # Shuffle the data
    return data[:reduced_size]

# Check for TPU availability
if not tf.config.list_logical_devices('TPU'):  # No TPU available
    print("TPU not available. Reducing dataset size for CPU/GPU usage...")
    data = reduce_dataset(data, percentage=config.REDUCTION_PERCENTAGE)
    print(f"Dataset reduced to {config.REDUCTION_PERCENTAGE}% of its original size: {len(data)} rows remaining.")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    data["review"], data["label"], test_size=0.2, random_state=42
)

def train_bert_model(X_train, y_train, X_val, y_val, learning_rate=5e-5, batch_size=32, weight_decay=None, clip_norm=None):
    """Trains a BERT model with optional Gradient Clipping and Weight Decay."""
    
    # Tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2, from_pt=False
    )

    # Tokenize the text data
    train_encodings = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=128, return_tensors="tf"
    )
    val_encodings = tokenizer(
        list(X_val), truncation=True, padding=True, max_length=128, return_tensors="tf"
    )

    # Convert data to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_encodings), tf.convert_to_tensor(y_train.values, dtype=tf.int32))
    ).shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(val_encodings), tf.convert_to_tensor(y_val.values, dtype=tf.int32))
    ).batch(batch_size)

    # Optimizer with optional Weight Decay
    if weight_decay is not None:
        optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optimizers.AdamW(learning_rate=learning_rate)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Training loop
    epochs = 3
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # Training
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                inputs, labels = batch
                outputs = model(inputs, training=True)
                loss = loss_fn(labels, outputs.logits)
                train_loss += loss
                train_accuracy.update_state(labels, outputs.logits)

            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)

            # Apply optional Gradient Clipping
            if clip_norm is not None:
                gradients = [tf.clip_by_norm(g, clip_norm) for g in gradients]

            # Apply optimizer updates
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(f"Training loss: {train_loss / len(train_dataset)}")
        print(f"Training accuracy: {train_accuracy.result().numpy()}")

        # Validation
        val_loss = 0
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for batch in val_dataset:
            inputs, labels = batch
            outputs = model(inputs, training=False)
            val_loss += loss_fn(labels, outputs.logits)
            val_accuracy.update_state(labels, outputs.logits)

        print(f"Validation loss: {val_loss / len(val_dataset)}")
        print(f"Validation accuracy: {val_accuracy.result().numpy()}")

        # Update best validation accuracy
        if val_accuracy.result().numpy() > best_val_accuracy:
            best_val_accuracy = val_accuracy.result().numpy()

    return best_val_accuracy, val_loss / len(val_dataset), model

# Test different learning rates and batch sizes
learning_rates = [1e-5, 2e-5, 5e-5]
batch_sizes = [16, 32, 64]

best_model = None
best_accuracy = 0.0
best_hyperparams = {}

# Store results for all runs
results = []

for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"\nTraining with learning rate: {lr} and batch size: {batch_size}")
        val_acc, val_loss, model = train_bert_model(
            X_train, y_train, X_val, y_val,
            learning_rate=lr, batch_size=batch_size,
            weight_decay=0.01, clip_norm=1.0  # Optional: Set to None to disable
        )
        
        # Store results
        results.append({
            "learning_rate": lr,
            "batch_size": batch_size,
            "val_accuracy": val_acc,
            "val_loss": val_loss.numpy()
        })

        # Check if this is the best model so far
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model
            best_hyperparams = {"learning_rate": lr, "batch_size": batch_size}

# Save results to CSV
with open(RESULTS_CSV_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["learning_rate", "batch_size", "val_accuracy", "val_loss"])
    for res in results:
        writer.writerow([res["learning_rate"], res["batch_size"], res["val_accuracy"], res["val_loss"]])

print(f"\nTraining results saved to {RESULTS_CSV_PATH}")

# Save the best model
if best_model:
    best_model.save_pretrained(OUTPUT_MODEL_PATH)
    print(f"\nBest Model saved with LR: {best_hyperparams['learning_rate']} and Batch Size: {best_hyperparams['batch_size']}")
 
    
    
    
    
    
    

















from src.analysis import perform_analysis, print_analysis_results
from src.visualization import create_sentiment_distribution_plot, create_wordcloud
from src.preprocess_data import load_data, clean_data, preprocess_data, save_processed_data
from src.train_model import train_bert_model
import os
import sys
import matplotlib.pyplot as plt

def on_close(event):
    """Handler function to close the program when the plot window is closed."""
    print("Closing the program as the visualization window was closed.")
    global visualizations_closed
    visualizations_closed = True  # Set flag to indicate that visualizations have been closed
    sys.exit(0)  # Make sure the program exits after the plot window is closed

def main():
    # Define file paths
    input_file = "data/raw/IMDB Dataset.csv"
    output_file = "data/processed/processed_data.csv"
    output_model_path = "models/bert_text_classifier"

    # Debugging: Check if the output directory exists, or create it
    print(f"Checking if output directory exists for processed data: {os.path.dirname(output_file)}")
    if not os.path.exists(os.path.dirname(output_file)):
        print(f"Creating directory: {os.path.dirname(output_file)}")
        os.makedirs(os.path.dirname(output_file))

    # Load and processed data
    data = load_data(input_file)
    data_processed = clean_data(data)

    # Preprocess data for training
    X_train, X_test, y_train, y_test = preprocess_data(data_processed)

    # Save processed data (only 'review' and 'label' columns)
    save_processed_data(data_processed, output_file)

    # Debugging: Verify if the data was loaded and processed correctly
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Perform exploratory data analysis (EDA)
    analysis_results = perform_analysis(data_processed)
    print_analysis_results(analysis_results)

    # Create visualizations
    create_sentiment_distribution_plot(data_processed)
    create_wordcloud(data_processed)

    # Remove 'sentiment' column after analysis and visualizations
    if 'sentiment' in data_processed.columns:
        data_processed = data_processed.drop(columns=['sentiment'])

    # Set up the on_close event handler for the plots
    plt.gcf().canvas.mpl_connect('close_event', on_close)

    # Keep the plots open for inspection
    print("Please review the visualizations. Closing now...")

    # Display the plots and wait for them to be closed
    plt.show()

    # Flag to check if visualizations are closed
    global visualizations_closed
    visualizations_closed = False

    # Train the model with various learning rates and batch sizes
    learning_rates = [1e-5, 2e-5, 5e-5]
    batch_sizes = [16, 32, 64]
    
    # Loop through learning rates and batch sizes until visualizations are closed
    for lr in learning_rates:
        for batch_size in batch_sizes:
            print(f"Training with learning rate: {lr} and batch size: {batch_size}")
            train_bert_model(X_train, y_train, X_test, y_test, output_model_path, learning_rate=lr, batch_size=batch_size)
            
            # Check if visualizations were closed and exit the loop if they were
            if visualizations_closed:
                print("Visualizations closed, stopping further training.")
                break
        if visualizations_closed:
            break

    print("Training process completed.")
    print("Program has ended successfully.")

if __name__ == "__main__":
    main()
    