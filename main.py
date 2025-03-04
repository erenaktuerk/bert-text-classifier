from src.analysis import perform_analysis, print_analysis_results
from src.visualization import create_sentiment_distribution_plot, create_wordcloud
from src.preprocess_data import load_data, clean_data, preprocess_data, save_processed_data
from src.train_model import train_bert_model
from src.augment_data import augment_texts
import os
import sys
import matplotlib.pyplot as plt
import config  # Import the central configuration

def on_close(event):
    """Exit the program when the visualization window is closed."""
    print("Closing the program as the visualization window was closed.")
    sys.exit(0)

def main():
    # Define file paths
    input_file = "data/raw/IMDB Dataset.csv"
    output_file = "data/processed/processed_data.csv"
    output_model_path = "models/bert_text_classifier"

    # Ensure the output directory for processed data exists
    output_dir = os.path.dirname(output_file)
    print(f"Checking if output directory exists for processed data: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"Creating directory: {output_dir}")
        os.makedirs(output_dir)

    # Load and clean raw data
    data = load_data(input_file)
    data_processed = clean_data(data)

    # Preprocess data (split into train/test)
    X_train, X_test, y_train, y_test = preprocess_data(data_processed)

    # Save processed data (only 'review' and 'label' columns)
    save_processed_data(data_processed, output_file)

    # Debug: Print shapes of training data
    print(f"X_train shape: {X_train.shape}, y_train length: {len(y_train)}")

    # Perform exploratory data analysis (EDA)
    analysis_results = perform_analysis(data_processed)
    print_analysis_results(analysis_results)

    # Create visualizations
    create_sentiment_distribution_plot(data_processed)
    create_wordcloud(data_processed)

    # Optional: Apply data augmentation to the training data
    if config.USE_AUGMENTATION:
        print("Applying data augmentation to the training set...")
        augmented_X_train = augment_texts(list(X_train), aug_factor=config.AUGMENT_FACTOR)
        # Duplicate each label for the original + augmented texts
        augmented_y_train = []
        for label in y_train:
            augmented_y_train.extend([label] * (config.AUGMENT_FACTOR + 1))
        X_train, y_train = augmented_X_train, augmented_y_train
        print(f"After augmentation: {len(X_train)} training examples.")

    # Set up the on_close event handler for plot windows
    plt.gcf().canvas.mpl_connect('close_event', on_close)
    print("Please review the visualizations. Closing now...")
    # This call will block until alle Plots geschlossen werden
    plt.show()

    # Train the model with hyperparameter tuning using the parameters from config.py
    best_model = None
    best_accuracy = 0.0
    best_hyperparams = {}
    results = []

    for lr in config.LEARNING_RATES:
        for batch_size in config.BATCH_SIZES:
            print(f"\nTraining with learning rate: {lr} and batch size: {batch_size}")
            val_acc, val_loss, model = train_bert_model(
                X_train, y_train, X_test, y_test,
                learning_rate=lr,
                batch_size=batch_size,
                weight_decay=config.WEIGHT_DECAY,
                clip_norm=config.CLIP_NORM,
                epochs=config.EPOCHS
            )
            results.append({
                "learning_rate": lr,
                "batch_size": batch_size,
                "val_accuracy": val_acc,
                "val_loss": val_loss.numpy() if hasattr(val_loss, "numpy") else val_loss
            })
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model = model
                best_hyperparams = {"learning_rate": lr, "batch_size": batch_size}

    # Print training summary
    print("\n### Training Summary ###")
    for res in results:
        print(f"LR: {res['learning_rate']}, Batch Size: {res['batch_size']}, Val Accuracy: {res['val_accuracy']:.4f}, Val Loss: {res['val_loss']:.4f}")

    # Save the best model
    if best_model:
        best_model.save_pretrained(output_model_path)
        print(f"\nBest Model saved with LR: {best_hyperparams['learning_rate']} and Batch Size: {best_hyperparams['batch_size']}")

    print("Training process completed.")
    print("Program has ended successfully.")

if __name__ == "__main__":
    main()