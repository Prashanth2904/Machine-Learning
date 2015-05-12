import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import math

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
	    
    return  (1 / (1 + math.exp(-z)))    

def preprocess():
	mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data    
	train_data = np.array([])
	train_label = np.array([])
	validation_data = np.array([])
	validation_label = np.array([])
	test_data = np.array([])
	test_label = np.array([])
	
	#Representations of 1 through 9 for true labels.
	label0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	label1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	label2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
	label3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	label4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
	label5 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
	label6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
	label7 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
	label8 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	label9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	
	#Import data from mat as array.
	#One for training set (60000) and one for testing set (10000).
	train_data = np.vstack((mat['train0'], mat['train1'], mat['train2'], mat['train3'], mat['train4'], mat['train5'], mat['train6'], mat['train7'], mat['train8'], mat['train9']))
	test_data = np.vstack((mat['test0'], mat['test1'], mat['test2'], mat['test3'], mat['test4'], mat['test5'], mat['test6'], mat['test7'], mat['test8'], mat['test9'], ))
	
	#Cast to float type.
	train_data = train_data.astype(float)
	test_data = test_data.astype(float)
	
	#Normalize dataset.
	train_data = train_data / 255
	test_data = test_data / 255
	
	#Create vector with true labels. Train is 60000 rows, test is 10000 rows.
	train_label = np.vstack((np.full((mat['train0'].shape[0], 10), label0, dtype = np.int), np.full((mat['train1'].shape[0], 10), label1, dtype = np.int), np.full((mat['train2'].shape[0], 10), label2, dtype = np.int), np.full((mat['train3'].shape[0], 10), label3, dtype = np.int), np.full((mat['train4'].shape[0], 10), label4, dtype = np.int), np.full((mat['train5'].shape[0], 10), label5, dtype = np.int), np.full((mat['train6'].shape[0], 10), label6, dtype = np.int), np.full((mat['train7'].shape[0], 10), label7, dtype = np.int), np.full((mat['train8'].shape[0], 10), label8, dtype = np.int), np.full((mat['train9'].shape[0], 10), label9, dtype = np.int)))
	test_label = np.vstack((np.full((mat['test0'].shape[0], 10), label0, dtype = np.int), np.full((mat['test1'].shape[0], 10), label1, dtype = np.int), np.full((mat['test2'].shape[0], 10), label2, dtype = np.int), np.full((mat['test3'].shape[0], 10), label3, dtype = np.int), np.full((mat['test4'].shape[0], 10), label4, dtype = np.int), np.full((mat['test5'].shape[0], 10), label5, dtype = np.int), np.full((mat['test6'].shape[0], 10), label6, dtype = np.int), np.full((mat['test7'].shape[0], 10), label7, dtype = np.int), np.full((mat['test8'].shape[0], 10), label8, dtype = np.int), np.full((mat['test9'].shape[0], 10), label9, dtype = np.int)))

	#Combine training and testing data arrays with the vectors that contain the true labels for the data.
	#E.g. Row 1 of train_data will be a 28 x 28 pixel picture of the number 2, The last 10 columns in the row now contain the true label for testing algorithm which is [0, 0, 1, 0 ..., 0].
	train_big = np.hstack((train_data, train_label))
	test_big = np.hstack((test_data, test_label))
	
	#Randomize rows.
	np.random.shuffle(train_big)
	np.random.shuffle(test_big)
	
	#Split array back into two arrays separating data from true labels.
	split_train = np.split(train_big, [784,794], 1)
	split_test = np.split(test_big, [784,794], 1)
	train_data = split_train[0]
	train_label = split_train[1]
	test_data = split_test[0]
	test_label = split_test[1]
	
	#Split data array into two training and validation arrays.
	val_data_split = np.split(train_data, [50000,60000])
	val_label_split = np.split(train_label, [50000,60000])	
	train_data = val_data_split[0]
	train_label = val_label_split[0]
	validation_data = val_data_split[1]
	validation_label = val_label_split[1]
	
	return train_data, train_label, validation_data, validation_label, test_data, test_label    

def nnObjFunction(params, *args):
	n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    
	w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
	w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
	obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
	#	
	obj_val_i = np.zeros((training_data.shape[0], 1))
	
	for i in range(0, train_data.shape[0]):		
		out = nPredict(w1, w2, train_data, i)
		
		for l in range(0, n_class):		
			obj_val_i[i] = obj_val_i[i] + ((train_label[l] * log(out[l])) + ((1 - train_label[l]) * log(1 - out[l])))
		
	obj_val_i * -1
	
	obj_val_j = np.sum(obj_val_i)
	
	w_jp = 0
	w_lj = 0
	
	for j in range(0, n_hidden):
		
		for p in range(0, n_input):
			w_jp = w_jp + (w1[j, p] * w1[j, p])
	for l in range(0, n_class):
		
		for j in range(0, n_hidden):
			w_lj = w_lj + (w2[l, j] * w2[l, j])
			
	obj_val = obj_val_j + ((lambdaval / (2 * train_data.shape[0])) * (w_jp + w_lj)	
	
	#Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
	#you would use code similar to the one below to create a flat array
	#obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
	obj_grad = np.array([])
	obj_grad = np.ones(())
	
	return (obj_val,obj_grad)

def nnPredict(w1, w2, data):
	#Bias node
	bias = np.ones((data.shape[0], 1))
	data = np.hstack((data, bias))
	
	aj = np.ones((w1.shape[0], (data.shape[0] + 1)))
	#Multiply row in data by row in w1
	for y in range(0, data.shape[0]):
		for x in range(0, w1.shape[0]):
			aj[x, y] = sum(data[y] * w1[x])
	
	y = 0
	x = 0
	zj = aj
	#Apply sigmoid function to hidden nodes.
	for y in range(0, aj.shape[1]):
		for x in range(0, aj.shape[0]):
			zj[x, y] = sigmoid(aj[x, y])
	
	x = 0
	y = 0
	bias = np.ones((1, zj.shape[1]))
	zj = np.vstack((zj, bias))
	bl = np.ones((w2.shape[0], (data.shape[0] + 1)))
	
	#Multiply column of output from z with row of w2
	for y in range(0, zj.shape[1]):
		for x in range(0, w2.shape[0]):
			bl[x, y] = sum(zj[:, y] * w2[x])
		
	x = 0
	y = 0
	ol = bl
	
	#Apply sigmoid function to output nodes
	for y in range(0, bl.shape[1]):
		for x in range(0, bl.shape[0]):
			ol[x, y] = sigmoid(bl[x, y])
	
	maxs = np.ones((ol.shape[1], 2))
	x = 0
	y = 0
	#Find max (predicted) value for each data instance.
	for x in range(0, ol.shape[1]):
		maxs[x, 0] = np.amax(ol[:, x])
		maxs[x, 1] = np.argmax(ol[:, x])
		
	labels = maxs
	
	return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
				   
# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

#print nnPredict(initial_w1, initial_w2, train_data)

# set the regularization hyper-parameter
lambdaval = 0.1;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')