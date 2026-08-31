# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 31-08-2026

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
# 1. Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


# 2. Read the BMW CSV file

df = pd.read_csv(r"C:\Users\admin\Downloads\bmw (1).csv")

print("BMW DATASET")
print("Shape:", df.shape)

print("\nFIRST FIVE ROWS:")
print(df.head())


# Use price column as the time series

data = df["price"].dropna().reset_index(drop=True)

print("\nPRICE DATA:")
print(data.head())


# 3. Plot the original time series

plt.figure(figsize=(12, 5))
plt.plot(data)

plt.title("BMW Car Price Time Series")
plt.xlabel("Observation")
plt.ylabel("Price")

plt.show()


# 4. ADF Test - Check Stationarity

adf_test = adfuller(data)

print("\nAUGMENTED DICKEY-FULLER TEST")
print("ADF Statistic:", adf_test[0])
print("p-value:", adf_test[1])

if adf_test[1] < 0.05:
    print("Result: The data is stationary")
else:
    print("Result: The data is not stationary")


# 5. ACF and PACF of original data

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(data, lags=30, ax=ax[0])
ax[0].set_title("ACF - BMW Price")

plot_pacf(data, lags=30, ax=ax[1])
ax[1].set_title("PACF - BMW Price")

plt.tight_layout()
plt.show()


# 6. Differencing to make the data stationary

diff_data = data.diff().dropna()

plt.figure(figsize=(12, 5))
plt.plot(diff_data)

plt.title("Differenced BMW Price Data")
plt.xlabel("Observation")
plt.ylabel("Differenced Price")

plt.show()


# 7. ADF Test after differencing

adf_diff = adfuller(diff_data)

print("\nADF TEST AFTER DIFFERENCING")
print("ADF Statistic:", adf_diff[0])
print("p-value:", adf_diff[1])

if adf_diff[1] < 0.05:
    print("Result: Differenced data is stationary")
else:
    print("Result: Differenced data is not stationary")


# 8. ACF and PACF after differencing

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(diff_data, lags=30, ax=ax[0])
ax[0].set_title("ACF - Differenced BMW Price")

plot_pacf(diff_data, lags=30, ax=ax[1])
ax[1].set_title("PACF - Differenced BMW Price")

plt.tight_layout()
plt.show()


# 9. Determine ARIMA parameters

# After first differencing:
# d = 1
# Use p = 1 and q = 1

p = 1
d = 1
q = 1

print("\nARIMA PARAMETERS")
print("p =", p)
print("d =", d)
print("q =", q)


# 10. Split data into training and testing

train_size = int(len(data) * 0.8)

train = data[:train_size]
test = data[train_size:]


# 11. Fit ARIMA model

model = ARIMA(
    train,
    order=(p, d, q)
)

model_fit = model.fit()

print("\nARIMA MODEL SUMMARY")
print(model_fit.summary())


# 12. Make predictions

predictions = model_fit.forecast(
    steps=len(test)
)

print("\nTIME SERIES PREDICTIONS")
print(predictions)


# 13. Evaluate model

mse = np.mean(
    (test.values - predictions.values) ** 2
)

rmse = np.sqrt(mse)

print("\nMODEL EVALUATION")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)


# 14. Plot actual vs predicted

plt.figure(figsize=(12, 5))

plt.plot(
    test.index,
    test.values,
    label="Actual Price"
)

plt.plot(
    test.index,
    predictions.values,
    label="Predicted Price"
)

plt.title("ARIMA(1,1,1) - BMW Price Prediction")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.legend()

plt.show()

# 15. Auto-fit ARIMA

from pmdarima import auto_arima

auto_model = auto_arima(
    train,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore"
)

print("\nAUTO ARIMA MODEL")
print(auto_model.summary())


# 16. Auto-ARIMA predictions

auto_predictions = auto_model.predict(
    n_periods=len(test)
)

print("\nAUTO ARIMA PREDICTIONS")
print(auto_predictions)


# 17. Evaluate Auto-ARIMA

auto_mse = np.mean(
    (test.values - auto_predictions) ** 2
)

auto_rmse = np.sqrt(auto_mse)

print("\nAUTO ARIMA MODEL EVALUATION")
print("Mean Squared Error:", auto_mse)
print("Root Mean Squared Error:", auto_rmse)


# 18. Plot Auto-ARIMA prediction

plt.figure(figsize=(12, 5))

plt.plot(
    test.index,
    test.values,
    label="Actual Price"
)

plt.plot(
    test.index,
    auto_predictions,
    label="Auto-ARIMA Prediction"
)

plt.title("Auto-ARIMA - BMW Price Prediction")
plt.xlabel("Observation")
plt.ylabel("Price")
plt.legend()

plt.show()
```
### OUTPUT:


<img width="925" height="742" alt="image" src="https://github.com/user-attachments/assets/2026c091-0982-489f-9a29-c9e4e60f8e53" />


<img width="888" height="635" alt="image" src="https://github.com/user-attachments/assets/f61e22bb-abd3-4508-888e-c4d5369472be" />


<img width="887" height="745" alt="image" src="https://github.com/user-attachments/assets/4c6570ce-f75d-48b4-a831-2d9a7ca71a6b" />

<img width="876" height="750" alt="image" src="https://github.com/user-attachments/assets/766acc35-3cb3-45cb-a9d6-0359578b4699" />

<img width="912" height="630" alt="image" src="https://github.com/user-attachments/assets/26e8649d-846b-4966-a38b-2e990500e543" />


### RESULT:
Thus the program run successfully based on the ARIMA model using python.
