Title: COVID-19 Presence Prediction Model Project through Symptoms
Author: Nam Jun Lee

Supported set of language: Python

Using libraries:

1. import pandas as pd
- This is responsible for loading the csv file and framing the data.

2. import matplotlib.pyplot as plt
- It serves to generate graphs as a visualization library.

3. import seaborn as sns
- This serves to create a heat map capable of checking the correlation coefficient.

4. from sklearn.linear_model import LogisticRegression
- This is a logistic regression algorithm used to predict the probability of categorical dependent variables.

5. from sklearn.metrics import classification_report
- It serves to measure the predictive quality of the classification algorithm.

6. from sklearn.model_selection import train_test_split
- It serves to divide the data into training and test data.

7. from sklearn.tree import DecisionTreeClassifier
- This is a decision tree algorithm that can be used for regression and classification problems.

8. from sklearn.svm import SVC
- It is a classifier linear support vector machine algorithm.

9. from sklearn.metrics import accuracy_score
- It serves to measure the accuracy of the classification algorithm.


Program Description:

This program retrieves Covid 19 data and checks the variables in each data. After generating a correlation plot, a preprocessing operation is performed to remove unnecessary data and convert the data types of variables. The training data are then divided into 70 percent and the rest is divided into test data. and create three models (Logistic Regression model, SVM model, Decision Tree model). After comparing the accuracy of each model, a graph is created and stored to see which variables are important through the most accurate model.


Result:

The results can be viewed in the terminal execution window, and the two plots can be found in the "Picture" directory.
First plot: Correlation plot of all variables for Covid variable.
Second plot: a plot of important variables obtained through an optimal model (Decision Tree model).