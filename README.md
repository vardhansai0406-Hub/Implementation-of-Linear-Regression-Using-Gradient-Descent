# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VARDHANSAI I 
RegisterNumber:25015695
*/
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

data=pd.read_csv("ex1.txt",header=None)

plt.scatter(data[0], data[1])

plt.xticks(np.arange(5,30,step=5))

plt.yticks(np.arange(-5,30, step=5))

plt.xlabel("Population of City(10,000s)")

plt.ylabel("Profit ($10,000)")

plt.title("Profit Prediction")

def computeCost(X, y, theta):

m=len(y)

h=X.dot(theta)

square_err=(h-y)**2

return 1/(2*m)*np.sum(square_err)

data_n=data.values

m=data_n[:,0].size

X=np.append(np.ones((m,1)), data_n[:,0].reshape(m,1), axis=1)

y=data_n[:,1].reshape(m,1)

theta=np.zeros((2,1))

computeCost(X,y, theta)

def gradientDescent (X,y, theta, alpha, num_iters):

m=len(y)

J_history=[] #empty list

for i in range(num_iters):

predictions=X.dot(theta)

error=np.dot(X.transpose(), (predictions-y))

descent=alpha*(1/m)*error

theta-=descent

J_history.append(computeCost(X,y, theta)) return theta,J_history

theta, J_history = gradientDescent (X, y, theta, 0.01,1500) print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)

plt.xlabel("Iteration")

plt.ylabel("$J(\Theta)$")

plt.title("Cost function using Gradient Descent")

plt.scatter(data[0], data[1]) x_value=[x for x in range(25)] y_value=[y*theta[1]+theta[0] for y in x_value] plt.plot(x_value,y_value, color="r") plt.xticks(np.arange(5,30, step=5)) plt.yticks(np.arange(-5,30, step=5)) plt.xlabel("Population of City(10,000s)") plt.ylabel("Profit ($10,000)") plt.title("Profit Prediction")
```

## Output:
<img width="977" height="907" alt="Screenshot 2026-02-12 172416" src="https://github.com/user-attachments/assets/cfdb37c6-b424-4ada-b5d7-54a4dc045471" />
<img width="990" height="544" alt="Screenshot 2026-02-12 172432" src="https://github.com/user-attachments/assets/751f27be-40dc-4d1e-9fff-f73744739726" />
<img width="994" height="686" alt="Screenshot 2026-02-12 172502" src="https://github.com/user-attachments/assets/24b0246d-3453-4ef1-92c2-110b943e943b" />
<img width="994" height="679" alt="Screenshot 2026-02-12 172513" src="https://github.com/user-attachments/assets/274a9552-e782-49b7-b536-d277fa2d6a37" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
