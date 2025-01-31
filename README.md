BERT Text Classifier

Overview

This project implements a BERT-based text classification model using TensorFlow and Keras. It processes raw text data, trains a classification model, evaluates performance, and visualizes results. The pipeline includes preprocessing, model training, evaluation, and result analysis.

Features
	•	Preprocessing: Cleans and tokenizes text data.
	•	BERT Model Training: Fine-tunes a BERT model for text classification.
	•	Evaluation: Computes loss and accuracy metrics on validation data.
	•	Visualization: Generates performance plots.
	•	TPU Support: Supports training on TPU for faster performance.

Installation

Ensure you have Python 3.8+ installed. Create a virtual environment and install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

For TPU training, a TPU-compatible environment (e.g., Google Colab) is recommended.

Project Structure

BERT Text Classifier
│── /data
│   │── /raw              # Raw input data
│   │── /cleaned_data     # Preprocessed data
│── /models               # Saved model checkpoints
│── /results
│   │── /plots            # Visualization outputs
│── /scripts
│   │── _init_.py
│   │── analysis.py       # Analyzes results
│   │── evaluate_model.py # Evaluates trained models
│   │── plot_results.py   # Generates performance plots
│   │── preprocess_data.py# Preprocesses raw data
│   │── train_model.py    # Trains BERT classifier
│   │── visualization.py  # Generates visual reports
│── /tf_env               # TensorFlow virtual environment
│── /venv                 # Python virtual environment
│── .gitignore            # Git ignored files
│── LICENSE               # Project license
│── main.py               # Main script for execution
│── README.md             # Documentation
│── requirements.txt      # Required dependencies

Usage

Preprocessing Data

Run preprocess_data.py to clean and tokenize raw text data:

python scripts/preprocess_data.py

	•	Input: /data/raw/
	•	Output: Processed data in /data/cleaned_data/

Training the Model

Train the BERT classifier:

python scripts/train_model.py

	•	Saves trained models in /models/
	•	Logs training metrics

TPU Support:
	•	If TPU is available, the full dataset is used.
	•	On CPU/GPU, the dataset is automatically reduced to 70%.
	•	If performance is still insufficient, manually adjust REDUCTION_FACTOR in train_model.py.

Evaluating the Model

Evaluate performance using:

python scripts/evaluate_model.py

	•	Displays training loss, validation loss, training accuracy, and validation accuracy.
	•	Saves results in /results/.

Generating Plots

Visualize results:

python scripts/plot_results.py

	•	Saves accuracy/loss plots in /results/plots/.

Full Pipeline Execution

To execute all steps sequentially:

python main.py

TPU Considerations
	•	If TPU is available, the full dataset is used.
	•	If TPU is not available, the dataset is reduced to 70% for CPU/GPU execution.
	•	If hardware cannot handle 70%, manually modify REDUCTION_FACTOR in:
	•	scripts/train_model.py
	•	scripts/evaluate_model.py

License

This project is licensed under the MIT License.