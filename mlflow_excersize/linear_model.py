import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

random_state = 40
np.random.seed(random_state)

# Load the diabetes dataset
boston = datasets.load_boston()
X_train, X_test, y_train, y_test = \
    train_test_split(boston.data, boston.target, random_state=random_state)

#lasso = Lasso(random_state=0, max_iter=10000)
ols = LinearRegression()

ols.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = ols.predict(X_test)

# The coefficients
print('Coefficients: \n', ols.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(y_test, y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")

#plt.xticks(())
# plt.yticks(())

plt.savefig('./mlflow_excersize/ols/test_scores.png')