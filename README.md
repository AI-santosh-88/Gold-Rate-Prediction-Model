
# TITLE : Gold Rate Prediction Model using Linear Regression

## Description:
This project develops a simple linear regression model to predict gold rates based on historical data. The model is trained using a dataset containing years and corresponding gold rates.  It aims to forecast future gold rates by identifying the linear relationship between the year and the price of gold. The project encompasses data loading, preprocessing, model training, prediction, evaluation, and model persistence.

## Responsibilities:

### 1.Data Loading and Preprocessing:
* Load the gold rate dataset from a CSV file ('Gold_Rate.csv') using pandas.
* Separate the dataset into features (Year - independent variable) and target (Gold Rate - dependent variable).
* Prepare the feature (x) as a NumPy array from the 'Year' column and the target (y) as a NumPy array from the 'Gold_Rate' column.
### 2.Data Splitting:
* Divide the dataset into training and testing sets using train_test_split from scikit-learn.
* Allocate 20% of the data for testing and 80% for training.
* Set random_state=0 for reproducible splitting.
### 3.Model Training:
* Initialize a Linear Regression model from scikit-learn.
* Train the Linear Regression model using the training data (x_train, y_train) using the fit() method.
### 4.Model Prediction:
* Predict gold rates for the test set (x_test) using the trained model and store the predictions in y_pred.
* Predict gold rates for the years 2026 and 2027 using the trained model.
### 5.Model Evaluation:
* Calculate and print the training score (R^2) to measure the model's fit on the training data.
* Calculate and print the testing score (R^2) to measure the model's generalization performance on unseen data.
* Calculate and print the Mean Squared Error (MSE) for both the training and testing sets to quantify the average squared difference between predicted and actual values.
### 6.Model Visualization:
* Create scatter plots to visualize the training and testing datasets, showing actual gold rates against the year.
* Overlay the regression line (predicted gold rates from the training data) on both the training and testing scatter plots to visually assess the model's fit.
### 7.Model Persistence:
* Save the trained Linear Regression model to a pickle file named 'gold_rate_pred_updated.pkl' using the pickle library.
* Print a confirmation message indicating that the model has been saved.

## Libraries Used:
#### * pandas: 
For data manipulation and CSV file reading (pd.read_csv, dataset.iloc).
#### * numpy: 
For numerical operations and array handling (np.array).
#### * scikit-learn (sklearn):
* model_selection.train_test_split: For splitting the dataset into training and testing sets.
* linear_model.LinearRegression: For implementing the linear regression model.
* metrics.mean_squared_error: For evaluating the model's performance using Mean Squared Error.
#### * pickle: 
For serializing and saving the trained model to a file (pickle.dump).
#### * matplotlib.pyplot:
For creating plots and visualizations (plt.scatter, plt.plot, plt.title, plt.xlabel, plt.ylabel, plt.show).

## Summary:
This project successfully implemented a linear regression model to predict gold rates based on the year.  Historical gold rate data was used to train the model, which was then evaluated on a separate test dataset. The model's performance was assessed using R-squared and Mean Squared Error, indicating its ability to capture the linear trend in gold rate changes over the years.  Predictions for future years (2026 and 2027) were generated, and the trained model was saved as a pickle file for future use without retraining. The visualizations provide a clear picture of the model's fit to both the training and testing data, demonstrating the linear relationship learned by the model.
