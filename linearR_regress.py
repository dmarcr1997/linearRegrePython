
import pandas as pd
import matplotlib.pyplot as mplot

file_name = input("Enter CSV file: ") #read in user input
dataset = pd.read_csv(file_name) #read in csv file 
X = dataset.iloc[:, :-1] #data set of first col
y = dataset.iloc[:, 1]  #data of second col
print(dataset)
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
#X_data contians 1st col
#Y_data contains 2nd col

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

graph_name = input("Enter ScatterPlot name: ")
x_name = input("Enter Name of X axis: ")
y_name = input("Enter Name of Y axis: ")
# Visualizing the Training set results
viz_train = mplot
viz_train.scatter(X_train, y_train, color='red')
viz_train.plot(X_train, regressor.predict(X_train), color='blue')
viz_train.title('{} (Training set)'.format(graph_name))
viz_train.xlabel(x_name)
viz_train.show(y_name)

# Visualizing the Test set results
viz_test = mplot
viz_test.scatter(X_test, y_test, color='red')
viz_test.plot(X_train, regressor.predict(X_train), color='blue')
viz_test.title('{} (Test set)'.format(graph_name))
viz_test.xlabel(x_name)
viz_test.ylabel(y_name)
viz_test.show()

choice = "y"
while(choice == "y" or choice == "Y"):
  regre_val = float(input("What value Would You like to find: "))
  y_pred = regressor.predict([[regre_val]])
  print("Here is your value {}".format(y_pred))
  choice = input("Would You like to predict more values(y/n): ")

print("Thank You")