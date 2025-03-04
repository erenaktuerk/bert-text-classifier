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

# Paths for processed data, model output and results CSV
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
OUTPUT_MODEL_PATH = "models/bert_text_classifier"
RESULTS_CSV_PATH = "results/training_results.csv"

# Ensure output directories exist
os.makedirs(OUTPUT_MODEL_PATH, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)

def reduce_dataset(data, percentage):
    """Reduce the dataset size to a specified percentage."""
    reduced_size = int(len(data) * (percentage / 100))
    data = shuffle(data, random_state=42)
    return data[:reduced_size]

# (Falls nötig, kann hier ebenfalls die Reduktion erfolgen – aktuell wird dies in main.py geregelt)

def train_bert_model(X_train, y_train, X_val, y_val, learning_rate=5e-5, batch_size=32, weight_decay=None, clip_norm=None, epochs=3):
    """Trains a BERT model with optional Gradient Clipping and Weight Decay."""
    
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, from_pt=False)
    
    # Tokenize training and validation texts
    train_encodings = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=128, return_tensors="tf"
    )
    val_encodings = tokenizer(
        list(X_val), truncation=True, padding=True, max_length=128, return_tensors="tf"
    )
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_encodings), tf.convert_to_tensor(y_train, dtype=tf.int32))
    ).shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(val_encodings), tf.convert_to_tensor(y_val, dtype=tf.int32))
    ).batch(batch_size)
    
    # Set up the optimizer (AdamW supports weight decay)
    if weight_decay is not None:
        optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optimizers.AdamW(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = 0
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for batch in train_dataset:
            with tf.GradientTape() as tape:
                inputs, labels = batch
                outputs = model(inputs, training=True)
                loss = loss_fn(labels, outputs.logits)
                train_loss += loss
                train_accuracy.update_state(labels, outputs.logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            if clip_norm is not None:
                gradients = [tf.clip_by_norm(g, clip_norm) for g in gradients]
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Training loss: {train_loss / len(train_dataset)}")
        print(f"Training accuracy: {train_accuracy.result().numpy()}")
        
        val_loss = 0
        val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for batch in val_dataset:
            inputs, labels = batch
            outputs = model(inputs, training=False)
            val_loss += loss_fn(labels, outputs.logits)
            val_accuracy.update_state(labels, outputs.logits)
        print(f"Validation loss: {val_loss / len(val_dataset)}")
        print(f"Validation accuracy: {val_accuracy.result().numpy()}")
        
        if val_accuracy.result().numpy() > best_val_accuracy:
            best_val_accuracy = val_accuracy.result().numpy()
    
    return best_val_accuracy, val_loss / len(val_dataset), model