# Heart_Disease_Prediction
# Table Of Content:
   ## Overview
   ## Dataset
   ## Dependencies
   ## Purpose
   ## Usage
   ## Model architecture
# Overview
This project will focus on predicting heart disease using neural networks. Based on attributes such as blood pressure, cholestoral levels, heart rate, and other characteristic attributes, patients will be classified according to varying degrees of coronary artery disease,which are utilized to train the ANN model for heart disease prediction.
# Dataset
This dataset contains patient data concerning heart disease diagnosis that was collected at several locations around the world. There are 76 attributes, including age, sex, resting blood pressure, cholestoral levels, echocardiogram data, exercise habits, and many others. To data, all published studies using this data focus on a subset of 14 attributes - so we will do the same. More specifically, we will use the data collected at the Cleveland Clinic Foundation.
To import the necessary data, we will use pandas' built in read_csv() function. 
# Dependencies
This project requires the following dependencies:
1.Python (version 3.x)
2.NumPy
3.pandas
4.scikit-learn
5.TensorFlow (or Keras, which is a high-level API for TensorFlow)
we can download the dependencies using !pip install dependencies name.
# Purpose 
1.Public Health Awareness
2.Preventive Medicine
3.Early Detection and Treatment
4.Improving Patient Care
5.Healthcare Policy and Planning
6.Quality of Life
# Usage
*1.Clone the Repository:
      git clone https://github.com/yourusername/heart-disease-prediction.git
      cd heart-disease-prediction
*2.Download the Dataset:
     Download the Cleveland Heart Disease dataset from the UCI repository and place it in the project directory.
*3.Preprocess Data:
     Ensure that the dataset is in the appropriate format.
     Perform any necessary preprocessing steps such as handling missing values, scaling features, and encoding categorical variables.
*4.Train the Model:
     Run the training script to train the ANN model on the preprocessed dataset.
*5.Evaluate the Model:
      Evaluate the trained model's performance on a separate test set to assess its accuracy and other relevant metrics.
Copy code
     python evaluate.py
*6.Make Predictions:
     Use the trained model to make predictions on new data.
Copy code
     python predict.py
# Model Architecture
The ANN model architecture consists of multiple layers of neurons with ReLU activation functions, followed by a softmax activation function in the output layer. The model is trained using the Adam optimizer with categorical cross-entropy loss.   
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Define a function to build the Keras model
def create_model():
    # Create model
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile model
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Print the model summary
print(model.summary())






