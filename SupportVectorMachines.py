import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def preprocess():
    """ 
     Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection
    """
    
    mat = loadmat('mnist_all.mat'); #loads the MAT object as a Dictionary
    
    n_feature = mat.get("train1").shape[1];
    n_sample = 0;
    for i in range(10):
        n_sample = n_sample + mat.get("train"+str(i)).shape[0];
    n_validation = 1000;
    n_train = n_sample - 10*n_validation;
    
    # Construct validation data
    validation_data = np.zeros((10*n_validation,n_feature));
    for i in range(10):
        validation_data[i*n_validation:(i+1)*n_validation,:] = mat.get("train"+str(i))[0:n_validation,:];
        
    # Construct validation label
    validation_label = np.ones((10*n_validation,1));
    for i in range(10):
        validation_label[i*n_validation:(i+1)*n_validation,:] = i*np.ones((n_validation,1));
    
    # Construct training data and label
    train_data = np.zeros((n_train,n_feature));
    train_label = np.zeros((n_train,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("train"+str(i)).shape[0];
        train_data[temp:temp+size_i-n_validation,:] = mat.get("train"+str(i))[n_validation:size_i,:];
        train_label[temp:temp+size_i-n_validation,:] = i*np.ones((size_i-n_validation,1));
        temp = temp+size_i-n_validation;
        
    # Construct test data and label
    n_test = 0;
    for i in range(10):
        n_test = n_test + mat.get("test"+str(i)).shape[0];
    test_data = np.zeros((n_test,n_feature));
    test_label = np.zeros((n_test,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("test"+str(i)).shape[0];
        test_data[temp:temp+size_i,:] = mat.get("test"+str(i));
        test_label[temp:temp+size_i,:] = i*np.ones((size_i,1));
        temp = temp + size_i;
    
    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis = 0);
    index = np.array([]);
    for i in range(n_feature):
        if(sigma[i] > 0.001):
            index = np.append(index, [i]);
    train_data = train_data[:,index.astype(int)];
    validation_data = validation_data[:,index.astype(int)];
    test_data = test_data[:,index.astype(int)];

    # Scale data to 0 and 1
    train_data = train_data/255.0;
    validation_data = validation_data/255.0;
    test_data = test_data/255.0;
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z));
    
def blrObjFunction(params, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector
    """
    labels = args[1];
    
    """
    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    n_data = args[0].shape[0];
    n_feature = args[0].shape[1];
    
    train_data_with_bias = np.hstack((np.ones((n_data,1)), args[0]));
    
    error = 0;
    error_grad = np.zeros((n_feature+1,1));
    wT = np.asmatrix(np.transpose(params));
    for n in range(n_data):
        xn = np.asmatrix(train_data_with_bias[n,:]).T;
        yn = sigmoid( np.dot(wT, xn) );
        tn = labels[n,0];
        
        error = error + (tn * np.log(float(yn))) + ( (1-tn) * np.log(float(1-yn)) );
        error_grad = error_grad + ( float(yn-tn) * xn );
  
    error = -error;
    error_grad = np.squeeze(np.asarray(error_grad));
    return error, error_grad

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    X_with_bias = np.hstack((np.ones((data.shape[0],1)), data));
    X_posteriors = np.zeros((data.shape[0],W.shape[1]));
    
    for i in range(X_with_bias.shape[0]):
        X_posteriors[i,:] = sigmoid(np.dot(np.transpose(W), X_with_bias[i,:]))
    
    label = np.zeros((data.shape[0],1));
    label = X_posteriors.argmax(axis=1).reshape(label.shape[0],1);

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

# number of classes
n_class = 10;

# number of training samples
n_train = train_data.shape[0];

# number of features
n_feature = train_data.shape[1];

T = np.zeros((n_train, n_class));

for i in range(n_class):
    T[:,i] = (train_label == i).astype(int).ravel();
    
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature+1, n_class));
initialWeights = np.zeros((n_feature+1,1));
opts = {'maxiter' : 50};

for i in range(n_class):
    labeli = T[:,i].reshape(n_train,1);
    args = train_data, labeli, i;
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:,i] = nn_params.x.reshape((n_feature+1,));
    
# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

####Linear Kernel####
print "Fitting linear kernel"

clf_linear = SVC(kernel = 'linear')
clf_linear.fit(train_data, np.ravel(train_label))

print "Done fitting, predicting linear kernel..."

print clf_linear.score(train_data, np.ravel(train_label))
print clf_linear.score(validation_data, np.ravel(validation_label))
print clf_linear.score(test_data, np.ravel(test_label))

####RBF Kernel####
print "Fitting rbf kernel, gamma of 1"

clf_rbfg = SVC(kernel = 'rbf', gamma = 1)
clf_rbfg.fit(train_data, np.ravel(train_label))

print "Done fitting, predicting rbf kernel..."

print clf_rbfg.score(train_data, np.ravel(train_label))
print clf_rbfg.score(validation_data, np.ravel(validation_label))
print clf_rbfg.score(test_data, np.ravel(test_label))

####RBF Kernel, default gamma####
print "Fitting rbf kernel, default gamma"

clf_rbf = SVC(kernel = 'rbf')
clf_rbf.fit(train_data, np.ravel(train_label))

print "Done fitting, predicting rbf kernel, default gamma..."

print clf_rbf.score(train_data, np.ravel(train_label))
print clf_rbf.score(validation_data, np.ravel(validation_label))
print clf_rbf.score(test_data, np.ravel(test_label))

####RBF Kernel, default gamma####
print "Fitting rbf with varying c values"

print "c = 1"
clf_c1 = SVC(kernel = 'rbf', C = 1)
clf_c1.fit(train_data, np.ravel(train_label))
print "c = 10"
clf_c10 = SVC(kernel = 'rbf', C = 10)
clf_c10.fit(train_data, np.ravel(train_label))
print "c = 20"
clf_c20 = SVC(kernel = 'rbf', C = 20)
clf_c20.fit(train_data, np.ravel(train_label))
print "c = 30"
clf_c30 = SVC(kernel = 'rbf', C = 30)
clf_c30.fit(train_data, np.ravel(train_label))
print "c = 40"
clf_c40 = SVC(kernel = 'rbf', C = 40)
clf_c40.fit(train_data, np.ravel(train_label))
print "c = 50"
clf_c50 = SVC(kernel = 'rbf', C = 50)
clf_c50.fit(train_data, np.ravel(train_label))
print "c = 60"
clf_c60 = SVC(kernel = 'rbf', C = 60)
clf_c60.fit(train_data, np.ravel(train_label))
print "c = 70"
clf_c70 = SVC(kernel = 'rbf', C = 70)
clf_c70.fit(train_data, np.ravel(train_label))
print "c = 80"
clf_c80 = SVC(kernel = 'rbf', C = 80)
clf_c80.fit(train_data, np.ravel(train_label))
print "c = 90"
clf_c90 = SVC(kernel = 'rbf', C = 90)
clf_c90.fit(train_data, np.ravel(train_label))
print "c = 100"
clf_c100 = SVC(kernel = 'rbf', C = 100)
clf_c100.fit(train_data, np.ravel(train_label))

print "Getting accuracies."
c1 = clf_c1.score(test_data, np.ravel(test_label))
c10 = clf_c10.score(test_data, np.ravel(test_label))
c20 = clf_c20.score(test_data, np.ravel(test_label))
c30 = clf_c30.score(test_data, np.ravel(test_label))
c40 = clf_c40.score(test_data, np.ravel(test_label))
c50 = clf_c50.score(test_data, np.ravel(test_label))
c60 = clf_c60.score(test_data, np.ravel(test_label))
c70 = clf_c70.score(test_data, np.ravel(test_label))
c80 = clf_c80.score(test_data, np.ravel(test_label))
c90 = clf_c90.score(test_data, np.ravel(test_label))
c100 = clf_c100.score(test_data, np.ravel(test_label))

plt.plot([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [c1, c10, c20, c30, c40, c50, c60, c70, c80, c90, c100])
plt.axis([0, 100, 0, 1])
plt.show()