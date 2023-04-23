import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from logistic_regression import CustomLogisticRegression, standardize

pd.options.mode.chained_assignment = None

data = load_breast_cancer(as_frame=True)

x = data.data[["worst concave points", "worst perimeter", "worst radius"]]
y = data.target

for col in x.columns:
    x.loc[:, col] = standardize(x.loc[:, [col]])

X_train, X_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=43
)

model = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)

model.fit_mse(X_train=X_train, y_train=y_train)
predictions_mse = model.predict(X_test=X_test)
accuracy_mse = accuracy_score(y_test.values, predictions_mse)

model.fit_log_loss(X_train=X_train, y_train=y_train)
predictions_ll = model.predict(X_test=X_test)
accuracy_ll = accuracy_score(y_test.values, predictions_ll)


print({"mse_accuracy": accuracy_mse, "logloss_accuracy": accuracy_ll})
