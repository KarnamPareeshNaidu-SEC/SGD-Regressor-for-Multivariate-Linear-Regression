# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
~~~
##1.Start the program and import the required libraries.
##2.Load the house dataset using Pandas.
##3.Select input features and target values from the dataset.
##4.Split the dataset into training and testing data.
##5.Train the model using SGD Regressor and predict the outputs.
##6.Display the predicted house price and number of occupants, then stop the program.
~~~
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: KARNAM PAREESH NAIDU
RegisterNumber: 212225230129 
*/
# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# Step 1: Create the dataset
# ------------------------------
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Marks_Scored':  [35, 40, 50, 55, 60, 65, 70, 75, 80, 85]
}

df = pd.DataFrame(data)
print("Dataset:")
print(df)

# ------------------------------
# Step 2: Split into X and Y
# ------------------------------
X = df[['Hours_Studied']]   # Feature (2D)
y = df['Marks_Scored']      # Target (1D)

# ------------------------------
# Step 3: Split data for training & testing
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 4: Create and train the model
# ------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred = model.predict(X_test)

# ------------------------------
# Step 6: Evaluate the model
# ------------------------------
print("\nModel Evaluation:")
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# ------------------------------
# Step 7: Visualize results
# ------------------------------
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Simple Linear Regression: Hours vs Marks')
plt.legend()
plt.show()

# ------------------------------
# Step 8: Predict for new data
# ------------------------------
hours = float(input("\nEnter number of study hours: "))
predicted_marks = model.predict([[hours]])
print(f"Predicted Marks for studying {hours} hours = {predicted_marks[0]:.2f}")

```

## Output:
<img width="802" height="412" alt="image" src="https://github.com/user-attachments/assets/176a7806-7407-4ef1-b38c-1e2837b0f738" />

<img width="1157" height="670" alt="image" src="https://github.com/user-attachments/assets/8d79e1fc-05b8-4ac8-b666-d1555fa47ead" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
