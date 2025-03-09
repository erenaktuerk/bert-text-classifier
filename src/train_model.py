# train_model.py
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import TFBertForSequenceClassification, BertTokenizer
from tensorflow.keras import optimizers
from sklearn.utils import shuffle
from src import config
from src.augment_data import augment_texts  # Import der Augmentierungsfunktion
import csv
import os

# Suppress TensorFlow warnings using the level from config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = config.TF_CPP_MIN_LOG_LEVEL
tf.get_logger().setLevel('ERROR')

# Paths
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
OUTPUT_MODEL_PATH = "models/bert_text_classifier"
RESULTS_CSV_PATH = "results/training_results.csv"

# Ensuring output directories exist
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
    data = shuffle(data, random_state=42)
    return data[:reduced_size]

#Augmentation function that duplicates labels accordingly
def apply_augmentation(X_train, y_train):
    """Apply data augmentation on the training dataset and replicate labels accordingly."""
    if config.USE_AUGMENTATION:
        print("Applying data augmentation...")
        augmented_X_train = []
        augmented_y_train = []
        # making sure the inputs are lists
        for text, label in zip(X_train.tolist(), y_train.tolist()):
            # keep OG text
            augmented_X_train.append(text)
            augmented_y_train.append(label)
            # run augmentation – expecting augment_texts returning a String-list
            aug_texts = augment_texts([text], aug_factor=config.AUGMENT_FACTOR)
            # skip the first Element, because it is the OG text
            for aug_text in aug_texts[1:]:
                # if the result is a list, take the first element
                if isinstance(aug_text, list):
                    aug_text = aug_text[0]
                augmented_X_train.append(aug_text)
                augmented_y_train.append(label)
        # debug-output: sampling augmented data
        #if len(augmented_X_train) > len(X_train):
         #   print(f"\nSample original text: {X_train.iloc[0]}")
          #  print(f"Sample augmented text: {augmented_X_train[len(X_train)]}")
           # print(f"Original label: {y_train.iloc[0]}")
            #print(f"Augmented label: {augmented_y_train[len(X_train)]}\n")
        #return augmented_X_train, augmented_y_train
    # if augmentation is deactivated, return OG data as list
    return X_train.tolist(), y_train.tolist()

# Check for TPU availability and reduce dataset if necessary
if not tf.config.list_logical_devices('TPU'):
    print("TPU not available. Reducing dataset size for CPU/GPU usage...")
    data = reduce_dataset(data, percentage=config.REDUCTION_PERCENTAGE)
    print(f"Dataset reduced to {config.REDUCTION_PERCENTAGE}% of its original size: {len(data)} rows remaining.")

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    data["review"], data["label"], test_size=0.2, random_state=42
)

# Apply augmentation to training data if enabled
X_train, y_train = apply_augmentation(X_train, y_train)

# Debug: Überprüfe Konsistenz von Trainingsdaten und Labels
print(f"Training data size: {len(X_train)}, Training labels size: {len(y_train)}")
if len(X_train) != len(y_train):
    raise ValueError("Mismatch between number of training texts and labels!")

# Initialize Tokenizer one time for reusability in training function
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def train_bert_model(X_train, y_train, X_val, y_val, learning_rate=5e-5, batch_size=32, weight_decay=None, clip_norm=None):
    """Trains a BERT model with optional Gradient Clipping and Weight Decay."""
    
    # Initialize a new model
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2, from_pt=False
    )

    # tokenize the text data
    train_encodings = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=128, return_tensors="tf"
    )
    val_encodings = tokenizer(
        list(X_val), truncation=True, padding=True, max_length=128, return_tensors="tf"
    )

    # create TensorFlow-Datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(train_encodings), tf.convert_to_tensor(y_train, dtype=tf.int32))
    ).shuffle(1000).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (dict(val_encodings), tf.convert_to_tensor(y_val, dtype=tf.int32))
    ).batch(batch_size)

    # Optimizer with optmal Weight Decay
    if weight_decay is not None:
        optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optimizers.AdamW(learning_rate=learning_rate)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # training loop
    epochs = config.EPOCHS
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

        # validation
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

# Hyperparameter search
learning_rates = [1e-5, 2e-5, 5e-5]
batch_sizes = [16, 32, 64]

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
            weight_decay=config.WEIGHT_DECAY, clip_norm=config.CLIP_NORM
        )
        
        results.append({
            "learning_rate": lr,
            "batch_size": batch_size,
            "val_accuracy": val_acc,
            "val_loss": val_loss.numpy()
        })
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = model
            best_hyperparams = {"learning_rate": lr, "batch_size": batch_size}

with open(RESULTS_CSV_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["learning_rate", "batch_size", "val_accuracy", "val_loss"])
    for res in results:
        writer.writerow([res["learning_rate"], res["batch_size"], res["val_accuracy"], res["val_loss"]])

print(f"\nTraining results saved to {RESULTS_CSV_PATH}")

if best_model:
    best_model.save_pretrained(OUTPUT_MODEL_PATH)
    print(f"\nBest Model saved with LR: {best_hyperparams['learning_rate']} and Batch Size: {best_hyperparams['batch_size']}")