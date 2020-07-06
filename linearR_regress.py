import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_name = raw_input("Enter CSV file: ") #read in user input
dataset = pd.read_csv(file_name) #read in csv file 
x_axis = dataset.iloc[:, :-1] #data set of first col
y_axis = dataset.iloc[:, 1]  #data of second col

from sklearn.model_selection import train_test_split
X_data, X_test, Y_data, Y_test = train_rest_split(x_axis, y_axis, test_size=1/3, random_state=0)

#X_data contians 1st col
#Y_data contains 2nd col

from sklearn.linear_model import LinearRegresion
regressor = LinearRegression()
regressor.fit(X_data, Y_data)
