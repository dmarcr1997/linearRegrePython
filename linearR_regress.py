
import pandas as pd
import matplotlib.pyplot as mplot

file_name = input("Enter CSV file: ") #read in user input
dataset = pd.read_csv(file_name) #read in csv file 
X = dataset.iloc[:, :-1] #data set of first col
y = dataset.iloc[:, 1]  #data of second col
print(dataset)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
#X_train contians 1st col
#Y_train contains 2nd col

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#get graph names/ xy name
graph_name = input("Enter ScatterPlot name: ")
x_name = input("Enter Name of X axis: ")
y_name = input("Enter Name of Y axis: ")

# Visualizing the Training set results

mplot.scatter(X_train, y_train, color='red') #create datapoint scatter plot
mplot.plot(X_train, regressor.predict(X_train), color='blue') #create linear line through pts
mplot.title('{} (Training set)'.format(graph_name)) #give graph a title
mplot.xlabel(x_name) #x axis name
mplot.ylabel(y_name) # y axis name
mplot.show() #print out graph

# Visualizing the Test set results

mplot.scatter(X_test, y_test, color='red')  #create test datapoint scatter plot
mplot.plot(X_train, regressor.predict(X_train), color='blue') #create linear line through pts
mplot.title('{} (Test set)'.format(graph_name)) #give test graph a title
mplot.xlabel(x_name) #x axis name
mplot.ylabel(y_name) #y axis name
mplot.show() #print out graph

choice = "y" 
while(choice == "y" or choice == "Y"): #while user wants to enter data loop
  regre_val = float(input("What {} Would You like to find: ".format(x_name))) #get value from user and convert to a float
  y_pred = regressor.predict([[regre_val]])[0] #send value to AI to get result
  print("Here is your {} {:.2f}".format(y_name, y_pred)) #print out result
  choice = input("Would You like to predict more values(y/n): ") #user input to try again

print("Thank You")