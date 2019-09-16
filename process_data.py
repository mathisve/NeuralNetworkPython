import os
import random 

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

def fetch_processed_data():
	#using https://www.kaggle.com/c/dogs-vs-cats
	#download and put the train and test1 folders in your main folder!
	train_dir = os.path.join(os.getcwd(), "train")
	test_dir = os.path.join(os.getcwd(), "test1")
	img_size = 35

	training_examples = os.listdir(train_dir)
	processed_training_examples = []
	processed_training_labels = []

	test_examples = os.listdir(test_dir)
	processed_testing_examples = []
	processed_testing_labels = []

	random.shuffle(training_examples)

	num_of_examples = 5000
	#num_of_examples = len(training_examples)


	for example in tqdm(training_examples[:num_of_examples]):
		if("dog" in example):
			classnum = 1
		else:
			classnum = 0

		temp = cv2.imread(os.path.join(train_dir, example), cv2.IMREAD_GRAYSCALE)
		img_array = cv2.resize(temp, (img_size, img_size))
		#Image.fromarray(img_array).show()
		#exit()
		img_array = img_array.flatten()
		#print(img_array.shape)
		img_array = img_array/255
		processed_training_examples.append(img_array)
		processed_training_labels.append(classnum)

	for example in tqdm(test_examples[:num_of_examples]):
		if("dog" in example):
			classnum = 1
		else:
			classnum = 0

		temp = cv2.imread(os.path.join(test_dir, example), cv2.IMREAD_GRAYSCALE)
		img_array = cv2.resize(temp, (img_size, img_size))
		img_array = img_array.flatten()
		#print(img_array.shape)
		img_array = img_array/255
		processed_testing_examples.append(img_array)
		processed_testing_labels.append(classnum)


	X_train = np.array(processed_training_examples).T
	Y_train = np.array(processed_training_labels).T

	X_test = np.array(processed_testing_examples).T
	Y_test = np.array(processed_testing_labels).T

	print("X_train shape: " + str(X_train.shape))
	print("Y_train shape: " + str(Y_train.shape))
	print("X_test shape: " + str(X_test.shape))
	print("Y_test shape: " + str(Y_test.shape))


	return {"X": X_train, "Y": Y_train}, {"X": X_test, "Y": Y_test}


	