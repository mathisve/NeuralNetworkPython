import process_data

import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

#using https://www.kaggle.com/c/dogs-vs-cats
#download and put the train and test1 folders in your main folder!
train, test = process_data.fetch_processed_data() #returns train {"X", "Y"} and test {"X", "Y"}

X, Y = train["X"], train["Y"]
m = X.shape[1]

iterations = 5000 #Epochs
lr = .01 #Learning rate

x = 1225
h1 = 700
h2 = 200
o = 1

def init_params():
	W1 = np.random.randn(h1, x) * 0.01
	b1 = np.zeros((h1, 1))

	W2 = np.random.randn(h2, h1) * 0.01
	b2 = np.zeros((h2, 1))

	W3 = np.random.randn(o, h2) * 0.01
	b3 = np.zeros((o, 1))

	return {"W1":W1, "b1":b1, "W2":W2, "b2":b2, "W3":W3, "b3":b3}

#Sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

#Forward propagaton
def forward_prop(params):
	Z1 = np.dot(params["W1"], X) + params["b1"]
	A1 = np.tanh(Z1)

	Z2 = np.dot(params["W2"], Z1) + params["b2"]
	A2 = np.tanh(Z2)

	Z3 = np.dot(params["W3"], Z2) + params["b3"]
	A3 = sigmoid(Z3)
	
	return {"Z1": Z1,"A1": A1,"Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}

def back_prop_and_update(params, cache):

	dZ3 = cache["A3"] - Y
	dW3	= (1/m)*np.dot(dZ3, cache["A2"].T)
	db3 = (1/m)*np.sum(dZ3, axis = 1, keepdims = True)

	dZ2 = cache["A2"] - Y
	dW2	= (1/m)*np.dot(dZ2, cache["A1"].T)
	db2 = (1/m)*np.sum(dZ2, axis = 1, keepdims = True)

	dZ1 = cache["A1"] - Y
	dW1	= (1/m)*np.dot(dZ1, X.T)
	db1 = (1/m)*np.sum(dZ1, axis = 1, keepdims = True)


	params["W1"] = params["W1"] - dW1 * lr
	params["b1"] = params["b1"] - db1 * lr

	params["W2"] = params["W2"] - dW2 * lr
	params["b2"] = params["b2"] - db2 * lr

	params["W3"] = params["W3"] - dW3 * lr
	params["b3"] = params["b3"] - db3 * lr

	return params	

def calc_accuracy(cache):
	yhat = np.round(cache["A3"])
	percentage = 1 - np.average(np.abs(yhat - Y))
	return percentage


params = init_params()
l1_cost, l2_cost, accuracy = [], [], []
for i in tqdm(range(iterations)):
	cache = forward_prop(params)
	params = back_prop_and_update(params, cache)
	accuracy.append(calc_accuracy(cache))
	l2_cost.append((1/np.sum(np.exp(cache["A3"] - Y)))*m)

	fig, ax1 = plt.subplots()
	#ax1.plot(l1_cost, label="l1", color="red")
	ax1.plot(l2_cost, label="l2 loss", color="blue")

	plt.legend()

	ax2 = ax1.twinx()
	ax2.plot(accuracy, label="accuracy", color="green")

	plt.legend()
	plt.pause(0.000001)
plt.show()
