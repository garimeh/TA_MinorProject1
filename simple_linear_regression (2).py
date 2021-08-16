import matplotlib.pyplot as plt
import pandas as pd

salary = pd.read_csv('salary_data.csv')
X = salary.iloc[:, :-1].values 
y = salary.iloc[:, 1].values 

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X_train, y_train)

y_pred = regressor1.predict(X_test)

visualise_train = plt
visualise_train.scatter(X_train, y_train, color='red')
visualise_train.plot(X_train, regressor1.predict(X_train), color='blue')
visualise_train.title('Salary VS Experience (Training set)')
visualise_train.xlabel('Year of Experience')
visualise_train.ylabel('Salary')
visualise_train.show()

visualise_test = plt
visualise_test.scatter(X_test, y_test, color='red')
visualise_test.plot(X_train, regressor1.predict(X_train), color='blue')
visualise_test.title('Salary VS Experience (Test set)')
visualise_test.xlabel('Year of Experience')
visualise_test.ylabel('Salary')
visualise_test.show()