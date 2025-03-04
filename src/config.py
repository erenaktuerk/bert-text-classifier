# Global configuration for the project

# Data processing
REDUCTION_PERCENTAGE = 0.05  # Reduction percentage for the dataset (e.g., 0.05 means 0.05% of the original data)

# Data augmentation
USE_AUGMENTATION = True      # Set to True to apply augmentation to the training data
AUGMENT_FACTOR = 1           # Number of augmented versions to generate per training example

# Training hyperparameters
LEARNING_RATES = [1e-5, 2e-5, 5e-5]  # List of learning rates to test
BATCH_SIZES = [16, 32, 64]           # List of batch sizes to test
WEIGHT_DECAY = 0.01                  # Weight decay for optimizer (if not None, AdamW will be used with this decay)
CLIP_NORM = 1.0                    # Gradient clipping norm (set to None, falls Gradient Clipping nicht gewünscht ist)
EPOCHS = 3                         # Number of training epochs