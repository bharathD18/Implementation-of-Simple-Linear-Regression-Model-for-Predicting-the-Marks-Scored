# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load dataset (Hours → X, Scores → Y).

2.Split data into training set and test set.

3.Train a Linear Regression model on training data.

4.Predict marks on test data.

5.Evaluate performance (MAE, MSE, RMSE).

6.Plot regression line with training and test data.

7.Use model to predict marks for new study hours.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: bharath.D
RegisterNumber: 212224240025

```
```
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset (Hours vs Scores)
data = {
    "Hours": [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7,
              7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4,
              2.7, 4.8, 3.8, 6.9, 7.8],
    "Scores": [21, 47, 27, 75, 30, 20, 88, 60, 81, 25,
               85, 62, 41, 42, 17, 95, 30, 24, 67, 69,
               30, 54, 35, 76, 86]
}
df = pd.DataFrame(data)

# Features and target
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict test results
Y_pred = regressor.predict(X_test)
print("Predicted:", Y_pred)
print("Actual:", Y_test)

# Training set plot
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Test set plot
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X, regressor.predict(X), color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Error metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:
<img width="560" height="47" alt="ml ep2" src="https://github.com/user-attachments/assets/15b7b171-9c17-48f5-81bf-ff67cc28a4fc" />
<img width="454" height="340" alt="ml ep 2" src="https://github.com/user-attachments/assets/b575ef58-b4c4-4e7c-b935-ab0f430755a6" />
<img width="897" height="666" alt="Screenshot 2025-09-02 113915" src="https://github.com/user-attachments/assets/96924eb8-bce5-4d76-8957-488540dcba7b" />
<img width="323" height="49" alt="ml" src="https://github.com/user-attachments/assets/3619a8d5-c731-4de3-9294-a3b89b243ba5" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
