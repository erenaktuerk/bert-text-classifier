import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Global reduction factor for dataset size (0.01 means 0.01% of original data)
REDUCTION_FACTOR = 1

def load_model(model_path):
    """
    Loads the fine-tuned BERT model and tokenizer.
    """
    model = TFBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Stelle sicher, dass das Modell die aktuellen TensorFlow-Funktionen nutzt
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    return model, tokenizer

def preprocess_data(tokenizer, texts, max_length=128):
    """
    Tokenizes the input texts and converts them into the format required by BERT.
    """
    encodings = tokenizer(
        texts.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='tf'
    )
    return encodings

def reduce_dataset(df):
    """
    Reduces the dataset by the globally defined percentage (REDUCTION_FACTOR).
    """
    reduced_size = int(len(df) * (REDUCTION_FACTOR / 100))  # Reducing by percentage
    df = df.sample(n=reduced_size, random_state=42).reset_index(drop=True)
    print(f"Dataset reduced to {len(df)} entries.")
    return df

def evaluate_model(model, tokenizer, test_data_path):
    """
    Evaluates the model on the test dataset and prints the metrics.
    """
    # Load test data
    test_data = pd.read_csv(test_data_path)
    
    # Print the label distribution in the test data
    print("Label distribution in the test dataset:")
    print(test_data['label'].value_counts())  # This will show the distribution of classes in the test dataset
    
    # Reduce the dataset if necessary
    test_data = reduce_dataset(test_data)

    texts = test_data['review']
    labels = test_data['label']

    # Preprocess the data
    encodings = preprocess_data(tokenizer, texts)

    # Model predictions
    logits = model(encodings).logits  # Offizielle Methode für den Zugriff auf die Logits
    predictions = tf.argmax(logits, axis=1).numpy()  # Moderne TensorFlow-Methode
    print("TEST: predicted labels distribution: ", np.bincount(predictions))

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == "__main__":
    model_path = 'models/bert_text_classifier'  # Path to the saved model
    test_data_path = 'data/processed/processed_data.csv'  # Path to the test dataset

    # Load model and tokenizer
    model, tokenizer = load_model(model_path)

    # Evaluate model (with automatic dataset reduction if needed)
    evaluate_model(model, tokenizer, test_data_path)