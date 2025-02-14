
# TITLE : Gold Rate Prediction Model using Linear Regression

## Description:
This project develops a simple linear regression model to predict gold rates based on historical data. The model is trained using a dataset containing years and corresponding gold rates.  It aims to forecast future gold rates by identifying the linear relationship between the year and the price of gold. The project encompasses data loading, preprocessing, model training, prediction, evaluation, and model persistence.

## Responsibilities:

### * Data Loading and Preprocessing:
* Load the gold rate dataset from a CSV file ('Gold_Rate.csv') using pandas.
* Separate the dataset into features (Year - independent variable) and target (Gold Rate - dependent variable).
* Prepare the feature (x) as a NumPy array from the 'Year' column and the target (y) as a NumPy array from the 'Gold_Rate' column.


