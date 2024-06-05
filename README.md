# Alphabet Soup Charity Funding Prediction
## Overview of the Analysis
The purpose of this analysis is to create a binary classifier that can predict whether applicants will be successful if funded by the nonprofit foundation Alphabet Soup. This will assist Alphabet Soup in selecting applicants with the best chance of success, thus ensuring that their funds are utilized effectively.

# Data Preprocessing
## Target and Feature Variables
  - Target Variable: IS_SUCCESSFUL - Indicates whether the funding was used effectively.
  - Feature Variables: All columns except EIN, NAME, and IS_SUCCESSFUL.
## Steps Taken in Data Preprocessing

Read the Data: The dataset was read from charity_data.csv into a Pandas DataFrame.

Identify and Drop Irrelevant Columns:
    -Dropped the EIN column as it is an identification column.
    -Dropped the NAME column initially, but it was used in the optimization notebook.

Unique Values Analysis:
    -Identified the number of unique values in each column.
    -For columns with more than 10 unique values, determined the number of data points for each unique value.

Handling Rare Categorical Variables:
    -Combined rare categorical values into a new category called Other to reduce the number of unique values.

Encoding Categorical Variables:
    -Used pd.get_dummies() to perform one-hot encoding on categorical variables.

Splitting the Data:
    -Split the preprocessed data into features array X and target array y.
    -Used train_test_split to split the data into training and testing datasets.

Scaling the Data:
    -Created a StandardScaler instance.
    -Fitted the scaler to the training data and transformed both the training and testing datasets.

## Model Compilation, Training, and Evaluation
## Initial Model

Structure:
Input Layer: Number of input features determined by the dataset.

Hidden Layers:
    -First hidden layer with 80 neurons and ReLU activation.
    -Second hidden layer with 30 neurons and ReLU activation.

Output Layer: Single neuron with sigmoid activation for binary classification.

## Compilation:
-   Loss Function: binary_crossentropy.
-   Optimizer: adam.
-   Metrics: accuracy.
## Training:
-   Epochs: 100.
## Evaluation:
-   Evaluated the model using the test data to determine loss and accuracy.
-   Saved the model to AlphabetSoupCharity.h5.

# Model Optimization
## Steps Taken for Optimization

## Input Data Adjustments:
Used the NAME column instead of dropping it. Including the NAME column provided more data for the model to learn from, as the dataset had over 19,000 unique names.

## Neural Network Adjustments:
Increased Neurons in Hidden Layers:
First hidden layer increased from 80 to 150 neurons.
Second hidden layer increased from 30 to 80 neurons.

## Added More Hidden Layers:
Added a third hidden layer with 30 neurons.

## Experimented with Different Activation Functions:
Used ReLU activation function for all hidden layers.

## Adjusted Number of Epochs:
Experimented with different numbers of epochs to find the optimal training duration.

# Result:
Trained the optimized model and evaluated its performance.
Saved the optimized model to AlphabetSoupCharity_Optimization.h5.

# Results
## Data Preprocessing
Target Variable: IS_SUCCESSFUL.
Feature Variables: All columns except EIN and IS_SUCCESSFUL.
Removed Columns: EIN and initially NAME (used in optimization).

## Model Details
## Neurons, Layers, and Activation Functions:
  Initial model had two hidden layers with 80 and 30 neurons respectively, using ReLU activation.
  Output layer used a sigmoid activation function.

## Target Performance:
  The initial model did not achieve the target performance of >75% accuracy.
  Optimization steps included adjusting input data and modifying the network architecture.

## Steps to Increase Model Performance:
1. Included the NAME column to provide more data for the model.
2. Added more neurons to the hidden layers.
3. Added additional hidden layers.
4. Experimented with different cut-off values for binning the NAME column
5. Adjusted the number of epochs to find the optimal training duration.

## Summary
- The initial deep learning model provided a solid baseline but did not meet the target accuracy. It gave us an accuracy of 0.72 and a loss of 0.55. The optimization process involved multiple changes to the network architecture and input data handling, which led to improvements in accuracy.

- Inclusion of NAME Column: Using the NAME column was based on the hypothesis that more data would improve model accuracy. Initial binning the NAME column with a cutoff value of "<700" improved accuracy slightly, but not to the desired outcome. Adjusting the cutoff value to "<250" improved accuracy further to 74.6%, but I still was not acheiving the 75%.  Finally, a cutoff value of "<50" achieved the desired accuracy of 0.76 and a loss of only 0.51. 

- Neural Network Adjustments: Adding another hidden layer and increasing neurons helped the model learn more complex patterns. The optimized model structure was 150, 80, 30, 1 neurons across layers.

Recommendations
For future improvements, exploring other machine learning models like Random Forest or Gradient Boosting could provide better performance due to their ability to handle complex relationships and interactions in the data. Using the "feature_importances" feature in the Random Forest Classifier library can help us predict which features are essential in the model's predictions, allowing you to adjust your model to have a better accuracy. 




By following these recommendations, Alphabet Soup can improve its funding allocation process, ensuring better outcomes for the funded projects.









