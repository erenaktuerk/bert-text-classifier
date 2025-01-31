import tensorflow as tf
import os
import pandas as pd
import csv
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from sklearn.utils import shuffle

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
CLEANED_DATA_PATH = "data/cleaned_data/cleaned_data.csv"
OUTPUT_MODEL_PATH = "models/bert_text_classifier"
RESULTS_CSV_PATH = "results/training_results.csv"

# Ensure output directories exist
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)

# Global variable for dataset reduction percentage
REDUCTION_PERCENTAGE = 70  # Set to 1 for 1% reduction

# Load cleaned data
data = pd.read_csv(CLEANED_DATA_PATH)

# Verify dataset structure
required_columns = ["review", "label"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The dataset must contain the following columns: {required_columns}")

# Subsampling function
def reduce_dataset(data, percentage=REDUCTION_PERCENTAGE):
    """Reduce the dataset size to a specified percentage."""
    reduced_size = int(len(data) * (percentage / 100))
    data = shuffle(data, random_state=42)  # Shuffle the data
    return data[:reduced_size]

# Check for TPU availability
if not tf.config.list_logical_devices('TPU'):  # No TPU available
    print("TPU not available. Reducing dataset size for CPU/GPU usage...")
    data = reduce_dataset(data, percentage=REDUCTION_PERCENTAGE)
    print(f"Dataset reduced to {REDUCTION_PERCENTAGE}% of its original size: {len(data)} rows remaining.")

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