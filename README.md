Project Overview: Gamma vs Hadrons Classification
This project aims to classify particles as either gamma rays or hadrons using data from the MAGIC gamma telescope. The dataset contains 20 parameters describing each particle event, such as length, width, size, concentration, asymmetry, and more.

Data Processing
The raw data is loaded and columns are named for clarity.
The target variable (class) is encoded: gamma (g) as 1, hadron (h) as 0.
Data is split into training, validation, and test sets.
Features are standardized using StandardScaler.
Oversampling is applied to the training set to address class imbalance.
Model Training
Several classification models were trained and evaluated:

K Nearest Neighbours (KNN)
Naive Bayes
Logistic Regression
Support Vector Machine (SVM)
Neural Network (TensorFlow/Keras)
Each model was trained on the processed data and evaluated using the test set. The neural network was tuned by cycling through different hyperparameters (number of nodes, dropout rate, batch size, learning rate, epochs) to find the configuration yielding the lowest validation loss.

Model Saving
All trained models are saved in the models folder for future use.
Plots showing feature distributions and training history (loss and accuracy) are saved in the reports folder.
Results
The best neural network configuration and its validation loss are reported. All models can be loaded and used for inference to predict whether a particle is a gamma ray or hadron based on its parameters.
