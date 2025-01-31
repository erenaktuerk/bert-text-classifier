import matplotlib.pyplot as plt
import pandas as pd
import os

# Paths
RESULTS_CSV_PATH = "results/training_results.csv"
PLOTS_OUTPUT_PATH = "results/plots"

# Ensure the output directory exists
os.makedirs(PLOTS_OUTPUT_PATH, exist_ok=True)

def plot_results(results_csv=RESULTS_CSV_PATH):
    """Plots training results (loss and accuracy) from a CSV file."""
    
    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"Results file not found: {results_csv}")

    # Load results
    df = pd.read_csv(results_csv)

    # Verify required columns
    required_columns = ["learning_rate", "batch_size", "val_accuracy", "val_loss"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")

    # Sort values for better plotting
    df = df.sort_values(by=["learning_rate", "batch_size"])

    # Plot Validation Accuracy
    plt.figure(figsize=(10, 5))
    for lr in df["learning_rate"].unique():
        subset = df[df["learning_rate"] == lr]
        plt.plot(subset["batch_size"], subset["val_accuracy"], marker="o", linestyle="--", label=f"LR: {lr}")

    plt.xlabel("Batch Size")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Batch Size")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(PLOTS_OUTPUT_PATH, "validation_accuracy.png"))
    plt.show()

    # Plot Validation Loss
    plt.figure(figsize=(10, 5))
    for lr in df["learning_rate"].unique():
        subset = df[df["learning_rate"] == lr]
        plt.plot(subset["batch_size"], subset["val_loss"], marker="o", linestyle="--", label=f"LR: {lr}")

    plt.xlabel("Batch Size")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss vs. Batch Size")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(PLOTS_OUTPUT_PATH, "validation_loss.png"))
    plt.show()

# Run the function if the script is executed directly
if __name__ == "__main__":
    plot_results()