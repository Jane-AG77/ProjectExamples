import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# get dataset
data = fetch_california_housing()
X, y = data.data, data.target

#split your data into training and testing set
# Use test_size = 0.2 (80% of data for training, 20% data for testing)

#got function from Linear Regression and more.ipynb
X_train, X_test, y_train_Big, y_test_Big = train_test_split(X, y, test_size = 0.2, random_state=42)


# normalize X data. 
# fit the normalizer on X_train, but apply the same to X_test as well 

scaler = StandardScaler()
X_train_Big = scaler.fit_transform(X_train)
X_test_Big = scaler.transform(X_test)

# 1.3 (Done for you): Fit a default Linear Regressor (no hyperparameters specified)
#   and report the training and testing MSE


sgd_reg = SGDRegressor()
sgd_reg.fit(X_train_Big, y_train_Big)
train_mse = mean_squared_error(y_train_Big, sgd_reg.predict(X_train_Big))
test_mse = mean_squared_error(y_test_Big, sgd_reg.predict(X_test_Big))
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state=42)

scaler = StandardScaler()
X_train_Small = scaler.fit_transform(X_train)
X_test_Small = scaler.transform(X_test)


gd_reg = SGDRegressor()
sgd_reg.fit(X_train_Small, y_train)
train_mse = mean_squared_error(y_train, sgd_reg.predict(X_train_Small))
test_mse = mean_squared_error(y_test, sgd_reg.predict(X_test_Small))
print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")

# Revert back to test_size = 0.2 for your experiments again

learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]

# Train a SGDRegressor on your training data, but for varying learning_rates
# use each of the values stated above
# for each learning rate, train a different SGD regressor and report the MSE of 
#   training and testing data

# Look up SGDRegressor API, but the way to fix learning rate is this:
# sgd_reg = SGDRegressor(learning_rate='constant', eta0=eta), where eta is your given learning_rate

for eta in learning_rates:
    sgd_reg = SGDRegressor(learning_rate = 'constant', eta0 = eta)
    sgd_reg.fit(X_train_Big, y_train_Big)
    train_mse = mean_squared_error(y_train_Big, sgd_reg.predict(X_train_Big))
    test_mse = mean_squared_error(y_test_Big, sgd_reg.predict(X_test_Big))
    print(f"eta0: {eta}")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Testing MSE: {test_mse:.4f}")


# I googled and got results using Google's AI Overview for the following questions:
# How to use StandardScaler: 
#   Got scaler = StandardScaler(), fit.transform() and transform()
#   lines 31 - 33 (modified from Linear Regression and more.ipynb)
