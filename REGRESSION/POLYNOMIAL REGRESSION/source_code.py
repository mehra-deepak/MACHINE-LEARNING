# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 22:21:32 2019

@author: Lenovo
"""

#importing the essential libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#splitting the dataset into training and testing set

#from sklearn.cross_validation import train_test_split
#X_train,X_test,y_train,y_test = train_test_split(X,y,X_test =


#fitting linear regression to the dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#fitting polimial regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)