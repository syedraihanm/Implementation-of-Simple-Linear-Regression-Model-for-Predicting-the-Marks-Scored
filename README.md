# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

Devoloped by : Syed Mohamed Raihan.M

Reg No: 212224240167

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and split it into features (Hours) and target (Scores).

2. Divide the data into training and testing sets using train_test_split().

3. Train a LinearRegression model on the training data.

4. Predict scores for the test set and visualize with scatter and line plots.

5. Evaluate model performance using MSE, MAE, and RMSE.
 

## Program:
```


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("student_scores.csv")

# Display the first and last few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nLast 5 rows of the dataset:")
print(df.tail())

# Separate the independent (X) and dependent (Y) variables
X = df.iloc[:, :-1].values  # 'Hours' column
Y = df.iloc[:, -1].values   # 'Scores' column

# Split the dataset into training and testing sets (1/3 for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Create and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the test set results
Y_pred = regressor.predict(X_test)

# Display predicted and actual values
print("\nPredicted values:", Y_pred)
print("Actual values:", Y_test)

# Plot the Training set results
plt.scatter(X_train, Y_train, color="red", label="Actual Scores (Train)")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Fitted Line")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.grid(True)
plt.show()

# Plot the Testing set results
plt.scatter(X_test, Y_test, color='green', label="Actual Scores (Test)")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print error metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print('\n--- Error Metrics ---')
print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("student_scores.csv")

# Display the first and last few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

print("\nLast 5 rows of the dataset:")
print(df.tail())

# Separate the independent (X) and dependent (Y) variables
X = df.iloc[:, :-1].values  # 'Hours' column
Y = df.iloc[:, -1].values   # 'Scores' column

# Split the dataset into training and testing sets (1/3 for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Create and train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the test set results
Y_pred = regressor.predict(X_test)

# Display predicted and actual values
print("\nPredicted values:", Y_pred)
print("Actual values:", Y_test)

# Plot the Training set results
plt.scatter(X_train, Y_train, color="red", label="Actual Scores (Train)")
plt.plot(X_train, regressor.predict(X_train), color="blue", label="Fitted Line")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.grid(True)
plt.show()

# Plot the Testing set results
plt.scatter(X_test, Y_test, color='green', label="Actual Scores (Test)")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Fitted Line")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores Achieved")
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print error metrics
mse = mean_squared_error(Y_test, Y_pred)
mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print('\n--- Error Metrics ---')
print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)



```

## Output:

![Screenshot 2025-04-30 141000](https://github.com/user-attachments/assets/58d0d711-005d-4cc7-86d8-097cebc95a5d)
![Screenshot 2025-04-30 141154](https://github.com/user-attachments/assets/52660918-1005-48fd-885f-54961ff42922)
![Screenshot 2025-04-30 141452](https://github.com/user-attachments/assets/fb0a518a-76c0-4ac7-920d-4c5b298eac51)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
