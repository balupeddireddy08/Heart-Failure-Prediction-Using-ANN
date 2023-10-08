# Exploratory Data Analysis (EDA) of Heart Failure Dataset and Heart-Failure-Prediction-Using-ANN

## Steps Taken:

### Importing Libraries:
I start by importing the necessary Python libraries such as Pandas, NumPy, Matplotlib, Seaborn, and warnings.
I also set up the Matplotlib to display plots inline.

### Importing the Dataset:
I load the Heart Failure Clinical Records dataset from a CSV file into a Pandas DataFrame.
Displayed the first few rows of the dataset using the head() method to get a glimpse of the data.

### Data Overview:
Used the describe() method to generate statistical summaries of numerical columns in the dataset.
Checked data types and non-null counts for each column using the info() method.
Verified that there are no missing values in any of the columns using the isnull().sum() method.

### Data Visualization:
Created various visualizations to explore the dataset.
Plotted a count plot to visualize the distribution of the target variable 'DEATH_EVENT,' which indicates whether a patient survived or deceased.
Visualized gender distribution among patients and how it relates to the target variable.
Explored the impact of high blood pressure on patient outcomes.
Investigated the effect of smoking habits on patient outcomes.
Generated box plots to visualize the distribution of numerical features.

### Statistical Analysis:
Calculated and displayed statistics like mean, kurtosis, and skewness for specific numerical columns ('serum_sodium' and 'age').
Plotted density distribution plots to visualize the distribution of these columns.

### Pandas Profiling:
Utilized the Pandas Profiling library to generate a comprehensive report summarizing the dataset's characteristics, including statistics, missing values, and correlations.

### Correlation Analysis:
Computed the correlation matrix to analyze the relationships between numerical features.
Visualized the correlation matrix using a heatmap to identify strong correlations.

### Pairplot:
Created a pairplot to visualize relationships between numerical features in scatter plots.


### Data Preprocessing and Model Training 

#### Outlier Detection using IQR Method:
Created a function iqr_method to detect outliers using the Interquartile Range (IQR) method for a specified column.
Detected outliers for the 'serum_sodium' column and displayed the corresponding rows.

#### Automated Handling of Outliers:
Created a function automated_handling_outliers to automatically handle outliers for all columns in the dataset.
The function identifies outlier columns and applies a Box-Cox transformation to handle them.
Displayed the columns that had outliers and were transformed.

#### Splitting the Dataset:
Split the preprocessed dataset into features (X) and the target variable (y).
Used the train_test_split function from Scikit-Learn to split the data into training and testing sets.

#### Data Scaling:
Applied Min-Max scaling to normalize the feature data.
Scaled both the training and testing sets.

#### Importing TensorFlow and Model Creation:
Imported necessary modules from TensorFlow/Keras.
Created a neural network model with three hidden layers and an output layer with sigmoid activation for binary classification.
Compiled the model with binary cross-entropy loss and the Adam optimizer.

#### Model Training:
Trained the model using the training data and validation data.
Monitored the validation loss to implement early stopping if the loss stopped improving.
Visualized the training and validation loss over epochs.

#### Model Evaluation:
Created a DataFrame df_y to store the actual and predicted values.
Applied a threshold of 0.5 to the predicted values to classify them as 0 or 1.
Calculated accuracy, precision, recall, and generated a classification report to evaluate model performance.

#### Visualization:
Plotted the validation loss with and without early stopping to visualize the effect of early stopping on model training.
Plotted the training loss with and without early stopping.

#### Conclusion:
Outliers in the dataset were detected and handled using the IQR method.
The dataset was split into training and testing sets.
Feature scaling was applied to normalize the data.
A neural network model was created and trained to predict heart failure.
Early stopping was implemented to prevent overfitting.
Model evaluation showed that the model achieved good accuracy, precision, and recall on the test data.
The visualizations demonstrate the impact of early stopping on the training process.
This code provides a comprehensive pipeline for preprocessing data, handling outliers, training a neural network model, and evaluating its performance for heart failure prediction.








