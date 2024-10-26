Vehicle Crash Prediction Model
Overview
This project implements a machine learning model to predict the number of crashes for different vehicle makes based on historical crash data. The model uses linear regression to establish relationships between vehicle manufacturers and crash frequencies.
Features

Data preprocessing and cleaning
Linear regression model implementation
Performance metrics calculation (MSE, RMSE, R²)
Visualization of actual vs predicted crashes
Easy-to-use prediction interface
Comprehensive performance analysis by vehicle make

Requirements
Copypython >= 3.7
pandas >= 1.2.0
numpy >= 1.19.0
scikit-learn >= 0.24.0
matplotlib >= 3.3.0
Installation

Clone the repository:

bashCopygit clone https://github.com/yourusername/vehicle-crash-prediction.git
cd vehicle-crash-prediction

Install required packages:

bashCopypip install pandas numpy scikit-learn matplotlib
Usage
Basic Usage
pythonCopy# Import the model
from crash_prediction_model import create_crash_prediction_model

# Prepare your data
data = [
    ['TOYOTA', 25973],
    ['HONDA', 21014],
    ['FORD', 18517],
    ['NISSAN', 9406],
    ['TOYT', 8841],
    # Add more makes and their crash counts
]

# Create and train the model
result = create_crash_prediction_model(data)

# Make predictions
make_to_predict = "TOYOTA"
prediction = result['predict_function'](make_to_predict)
print(f"Predicted crashes for {make_to_predict}: {prediction}")
Advanced Usage
pythonCopy# Get model metrics
print("\nModel Metrics:")
print(f"R² Score: {result['metrics']['r2']:.4f}")
print(f"Root Mean Square Error: {result['metrics']['rmse']:.2f}")
print(f"Mean Square Error: {result['metrics']['mse']:.2f}")

# View performance comparison
print(result['performance_df'].head())

# Create visualization
from crash_prediction_model import plot_actual_vs_predicted
plot_actual_vs_predicted(result['performance_df'])
Data Format
The input data should be a list of lists or a DataFrame with two columns:

Vehicle Make (string)
Number of Crashes (integer)

Example:
pythonCopydata = [
    ['MAKE_NAME', NUMBER_OF_CRASHES],
    ['TOYOTA', 25973],
    ['HONDA', 21014],
    # ...
]
Model Details
Preprocessing

Removes leading/trailing spaces from make names
Converts make names to uppercase
Encodes categorical makes to numerical values
Splits data into training (80%) and testing (20%) sets

Model Architecture

Uses scikit-learn's LinearRegression
Features: Encoded vehicle makes
Target: Number of crashes

Performance Metrics

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R² Score (Coefficient of Determination)

Output Example
pythonCopyModel Metrics:
R² Score: 0.8532
Root Mean Square Error: 1234.56
Mean Square Error: 1524123.45

Top Makes - Actual vs Predicted Crashes:
    make  actual_crashes  predicted_crashes
0  TOYOTA         25973           24521.34
1   HONDA         21014           20876.12
2    FORD         18517           18932.45
Limitations

The model assumes a linear relationship between vehicle makes and crashes
Predictions are based solely on vehicle make
Does not account for other potential factors such as:

Vehicle age
Model year
Geographic location
Weather conditions
Driver demographics



Future Improvements

Add more features (vehicle age, model year, etc.)
Implement more sophisticated models (Random Forest, XGBoost)
Add cross-validation
Include feature importance analysis
Add time-series analysis capabilities

License
This project is licensed under the MIT License - see the LICENSE.md file for details
