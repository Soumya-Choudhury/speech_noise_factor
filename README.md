# speech_noise_factor
# Import necessary libraries and modules
The first section imports the required libraries and modules. Notable libraries include os for interacting with the operating system, librosa for audio processing, numpy for numerical operations, svm from sklearn for Support Vector Machine, train_test_split for splitting the 
data, accuracy_score for evaluating model performance, and matplotlib.pyplot for plotting. The line np.random.seed(42) sets the seed for the NumPy random number generator to ensure reproducibility.

# Function Definitions
1)extract_mfcc: Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from an audio signal.
2)extract_lfcc: Extracts Log-Frequency Cepstral Coefficients (LFCCs) from an audio signal.
3)add_noise_by_factor: Adds random noise to an audio signal based on a given factor.

# Data Preparation and Train/Test Split
This section prepares the data by collecting file paths and corresponding labels from subdirectories. It then splits the data into training and testing sets using train_test_split.

# Feature Extraction for Training Data
# The model is trained only once, and the performance is tested with increasing noise factors
This section extracts features from the training data using MFCC and LFCC functions. It then trains a Support Vector Machine (SVM) classifier using the extracted features.

# Function to Test Model Performance with and without Noise
This section defines a function (test_model_with_and_without_noise) to evaluate the model's performance with different noise levels. It tests the SVM classifier with and without noise, recording and printing the accuracies.
Here, the script includes 0 in the noise factors for baseline testing and tests the SVM model with and without noise.

# Train the Random Forest Model
This section imports the RandomForestClassifier from sklearn.ensemble and trains a Random Forest classifier using the extracted features from the training data.
