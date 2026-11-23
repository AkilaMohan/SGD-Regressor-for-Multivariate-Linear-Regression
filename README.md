# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import NumPy, SGDRegressor, MultiOutputRegressor and StandardScaler.
2. Create the input data (X) and output data (Y).
3. Train the SGD regression model using the fit() function.
4. 1. Predict the house price and number of occupants using the predict() function and display the results.

## Program:
```

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = np.array([[1000, 2], [1500, 3], [2000, 4], [2500, 5], [3000, 6]])
Y = np.array([[20, 2], [30, 3], [40, 4], [50, 5], [60, 6]])

model = make_pipeline(
    StandardScaler(),
    MultiOutputRegressor(SGDRegressor(max_iter=10000, tol=1e-4, random_state=42))
)

model.fit(X, Y)

prediction = model.predict([[1800, 4]])

print("Predicted House Price:", prediction[0][0])
print("Predicted Number of Occupants:", prediction[0][1])

```

## Output:
<img width="350" height="62" alt="image" src="https://github.com/user-attachments/assets/7799c4b9-0f88-4bcb-a889-5753dde46d9f" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
