# Global configuration for the project

# Data processing
REDUCTION_PERCENTAGE = 0.06  # Reduction percentage for the dataset (e.g., 0.1 means 0.1% of the original data)

# Data augmentation
USE_AUGMENTATION = True     # Set to True to apply augmentation to the training data
AUGMENT_FACTOR = 1           # Number of augmented versions to generate per training example

# Training hyperparameters
LEARNING_RATES = [1e-5, 2e-5, 5e-5]  # List of learning rates to test
BATCH_SIZES = [16, 32, 64]           # List of batch sizes to test
WEIGHT_DECAY = 0.01                  # Weight decay for optimizer
CLIP_NORM = 1.0                    # Gradient clipping norm (set to None if not desired)
EPOCHS = 3                         # Number of training epochs

# TensorFlow settings
TF_CPP_MIN_LOG_LEVEL = "3"         # TensorFlow log level