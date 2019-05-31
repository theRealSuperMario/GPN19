import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Import mlflow
import mlflow

mlflow.set_tracking_uri('./mlflow_excersize/mlflow')
mlflow.set_experiment('diabetes')

random_seed = 40

with mlflow.start_run():
      np.random.seed(random_seed)
      mlflow.log_param('random_seed', random_seed)

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
      mse = mean_squared_error(y_test, y_pred)
      # The mean squared error
      print("Mean squared error: %.2f"
            % mse)
      r2 = r2_score(y_test, y_pred)
      # Explained variance score: 1 is perfect prediction
      print('Variance score: %.2f' % r2)

      mlflow.log_metric("MSE", mse)
      mlflow.log_metric("r2", r2)

      # Plot outputs
      plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
      plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

      plt.xticks(())
      plt.yticks(())

      img_path = './mlflow_excersize/ols/test_scores.png'
      plt.savefig(img_path)
      mlflow.log_artifact(img_path, artifact_path="imgs")