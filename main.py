from scripts.analysis import perform_analysis, print_analysis_results
from scripts.visualization import create_sentiment_distribution_plot, create_wordcloud
from scripts.preprocess_data import load_data, clean_data, preprocess_data, save_cleaned_data
from scripts.train_model import train_bert_model
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
    output_file = "data/cleaned_data/cleaned_data.csv"
    output_model_path = "models/bert_text_classifier"

    # Debugging: Check if the output directory exists, or create it
    print(f"Checking if output directory exists for cleaned data: {os.path.dirname(output_file)}")
    if not os.path.exists(os.path.dirname(output_file)):
        print(f"Creating directory: {os.path.dirname(output_file)}")
        os.makedirs(os.path.dirname(output_file))

    # Load and clean data
    data = load_data(input_file)
    data_cleaned = clean_data(data)

    # Preprocess data for training
    X_train, X_test, y_train, y_test = preprocess_data(data_cleaned)

    # Save cleaned data (only 'review' and 'label' columns)
    save_cleaned_data(data_cleaned, output_file)

    # Debugging: Verify if the data was loaded and processed correctly
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Perform exploratory data analysis (EDA)
    analysis_results = perform_analysis(data_cleaned)
    print_analysis_results(analysis_results)

    # Create visualizations
    create_sentiment_distribution_plot(data_cleaned)
    create_wordcloud(data_cleaned)

    # Remove 'sentiment' column after analysis and visualizations
    if 'sentiment' in data_cleaned.columns:
        data_cleaned = data_cleaned.drop(columns=['sentiment'])

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