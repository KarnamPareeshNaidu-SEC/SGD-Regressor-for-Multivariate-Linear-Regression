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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

data = {
    'Size': [1000, 1200, 1500, 1800, 2000],
    'Bedrooms': [2, 2, 3, 3, 4],
    'Price': [300000, 350000, 400000, 450000, 500000],
    'Occupants': [2, 3, 4, 5, 6]
}

df = pd.DataFrame(data)

X = df[['Size', 'Bedrooms']]

y = df[['Price', 'Occupants']]

model = MultiOutputRegressor(SGDRegressor())

model.fit(X, y)

prediction = model.predict([[1600, 3]])

print("Predicted Price:", prediction[0][0])
print("Predicted Occupants:", prediction[0][1])

plt.scatter(df['Size'], df['Price'])

plt.plot(df['Size'], model.predict(X)[:,0])

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Price Prediction")

plt.show()

plt.scatter(df['Size'], df['Occupants'])

plt.plot(df['Size'], model.predict(X)[:,1])

plt.xlabel("House Size")
plt.ylabel("Occupants")
plt.title("Occupants Prediction")

plt.show()
print("Trained Weights (including intercept):", weights)

# ------------------------------
# Step 5: Make predictions
# ------------------------------
y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)

```

## Output:
<img width="1125" height="649" alt="image" src="https://github.com/user-attachments/assets/3280e033-b81d-472b-841a-33c2e512168e" />

<img width="1008" height="627" alt="image" src="https://github.com/user-attachments/assets/db564f36-c5e2-4230-b4a4-d718920d07b4" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
