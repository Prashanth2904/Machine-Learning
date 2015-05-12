import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import math
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    d = X.shape[1]
    k = np.unique(y).shape[0]
    means = np.zeros((d, k)) #here 2,5
    # covmat - A single d x d learnt covariance matrix
    covmat = np.zeros((d, d)) #here 2,2
    
    # IMPLEMENT THIS METHOD
    C = np.hstack((y,X))
    C1 = C[C[:,0] == 1,:]
    CM1 = C1.mean(axis=0)
    C2 = C[C[:,0] == 2,:]
    CM2 = C2.mean(axis=0)
    C3 = C[C[:,0] == 3,:]
    CM3 = C3.mean(axis=0)
    C4 = C[C[:,0] == 4,:]
    CM4 = C4.mean(axis=0)
    C5 = C[C[:,0] == 5,:]
    CM5 = C5.mean(axis=0)
    #np.vstack((C1,C2,C3,C4,C5)) #Gives means for every class, try printing
    means_with_class = np.vstack((CM1,CM2,CM3,CM4,CM5))
    means = np.split(means_with_class.T, (1, d+1))[1]
      
    covmat1 = np.cov(np.split(C1,(1,3),axis=1)[1], rowvar=0) * C1.shape[0]/C.shape[0]
    covmat2 = np.cov(np.split(C2,(1,3),axis=1)[1], rowvar=0) * C2.shape[0]/C.shape[0]
    covmat3 = np.cov(np.split(C3,(1,3),axis=1)[1], rowvar=0) * C3.shape[0]/C.shape[0]
    covmat4 = np.cov(np.split(C4,(1,3),axis=1)[1], rowvar=0) * C4.shape[0]/C.shape[0]
    covmat5 = np.cov(np.split(C5,(1,3),axis=1)[1], rowvar=0) * C5.shape[0]/C.shape[0]
     
    covmat=covmat1+covmat2+covmat3+covmat4+covmat5
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    d = X.shape[1]
    k = np.unique(y).shape[0]
    means = np.zeros((d, k)) #here 2,5
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    covmats = np.empty( (1, k), dtype=np.ndarray )
    covmats.fill(np.zeros((d,d)))
    
    # IMPLEMENT THIS METHOD
    C = np.hstack((y,X))
    C1 = C[C[:,0] == 1,:].mean(axis=0)
    C2 = C[C[:,0] == 2,:].mean(axis=0)
    C3 = C[C[:,0] == 3,:].mean(axis=0)
    C4 = C[C[:,0] == 4,:].mean(axis=0)
    C5 = C[C[:,0] == 5,:].mean(axis=0)
    
    means_with_class = np.vstack((C1,C2,C3,C4,C5))
    means = np.split(means_with_class.T, (1, d+1))[1]
    
    for i in range(0, k):
        CC = C[C[:,0] == (i+1),:]
        covmats[0][i] = np.cov(np.split(CC,(1,3),axis=1)[1], rowvar=0) * CC.shape[0]/C.shape[0]
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    N = Xtest.shape[0]
    d = Xtest.shape[1]
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    count = 0.0
    prior1 = np.double(ytest[ytest[:,0]==1,:].shape[0])/ytest.shape[0]
    prior2 = np.double(ytest[ytest[:,0]==2,:].shape[0])/ytest.shape[0]
    prior3 = np.double(ytest[ytest[:,0]==3,:].shape[0])/ytest.shape[0]
    prior4 = np.double(ytest[ytest[:,0]==4,:].shape[0])/ytest.shape[0]
    prior5 = np.double(ytest[ytest[:,0]==5,:].shape[0])/ytest.shape[0]
    
    l_deno = (2*np.pi) * np.sqrt( np.linalg.det(np.asmatrix(covmat)) )
    merge = np.hstack((Xtest,ytest))
    
    colors = np.array(('g','black','r','m','y'))
    markers = np.array(('^','o','s','v','*'))
    
    plt.subplot(2, 1, 1)
    plt.title('LDA Test')
    plt.ylabel('x2')
    for i,entry in enumerate(merge):
        posterior1 = prior1 * (getLNumo(entry, means[:,0], covmat)/l_deno)[0][0]
        posterior2 = prior2 * (getLNumo(entry, means[:,1], covmat)/l_deno)[0][0]
        posterior3 = prior3 * (getLNumo(entry, means[:,2], covmat)/l_deno)[0][0]
        posterior4 = prior4 * (getLNumo(entry, means[:,3], covmat)/l_deno)[0][0]
        posterior5 = prior5 * (getLNumo(entry, means[:,4], covmat)/l_deno)[0][0]
        a = np.array([posterior1,posterior2,posterior3,posterior4,posterior5])
        if( (entry[2]==(np.argmax(a)+1)) ):
            count+=1
        plt.scatter(entry[0], entry[1], s=50, marker=markers[np.argmax(a)], color=colors[np.argmax(a)])
            
    return (count/merge.shape[0])*100

def getLNumo(xEntry, mEntry, covmat):
    part0 = np.double(-1)/2
    x = np.matrix((np.split(xEntry,(0,2))[1]))
    u = np.matrix(mEntry)
    part1 = (x-u)
    part2 = np.matrix(covmat).I
    part3 = part1.T   
    return np.exp(part0 * np.dot(np.dot(part1, part2), part3))
	
def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    N = Xtest.shape[0]
    d = Xtest.shape[1]
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    colors = np.array(('g','black','r','m','y'))
    markers = np.array(('^','o','s','v','*'))
    
    count = 0.0
    prior1 = np.double(ytest[ytest[:,0]==1,:].shape[0])/ytest.shape[0]
    prior2 = np.double(ytest[ytest[:,0]==2,:].shape[0])/ytest.shape[0]
    prior3 = np.double(ytest[ytest[:,0]==3,:].shape[0])/ytest.shape[0]
    prior4 = np.double(ytest[ytest[:,0]==4,:].shape[0])/ytest.shape[0]
    prior5 = np.double(ytest[ytest[:,0]==5,:].shape[0])/ytest.shape[0]
    
    l_deno1 = (2*np.pi) * np.sqrt( np.linalg.det(np.asmatrix(covmats[0,0])) )
    l_deno2 = (2*np.pi) * np.sqrt( np.linalg.det(np.asmatrix(covmats[0,1])) )
    l_deno3 = (2*np.pi) * np.sqrt( np.linalg.det(np.asmatrix(covmats[0,2])) )
    l_deno4 = (2*np.pi) * np.sqrt( np.linalg.det(np.asmatrix(covmats[0,2])) )
    l_deno5 = (2*np.pi) * np.sqrt( np.linalg.det(np.asmatrix(covmats[0,4])) )
    merge = np.hstack((Xtest,ytest))
    
    plt.subplot(2, 1, 2)
    plt.title('QDA Test')
    plt.ylabel('x2')
    for i,entry in enumerate(merge):
        posterior1 = prior1 * (getLNumo(entry, means[:,0], covmats[0,0])/l_deno1)
        posterior2 = prior2 * (getLNumo(entry, means[:,1], covmats[0,1])/l_deno2)
        posterior3 = prior3 * (getLNumo(entry, means[:,2], covmats[0,2])/l_deno3)
        posterior4 = prior4 * (getLNumo(entry, means[:,3], covmats[0,3])/l_deno4)
        posterior5 = prior5 * (getLNumo(entry, means[:,4], covmats[0,4])/l_deno5)
        a = np.array([posterior1,posterior2,posterior3,posterior4,posterior5])
        if( (entry[2]==(np.argmax(a)+1))):
            count+=1
        plt.scatter(entry[0], entry[1], s=50, marker=markers[np.argmax(a)], color=colors[np.argmax(a)])
            
    return (count/merge.shape[0])*100

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD 
    
	p1 = np.dot(X.T, X)
	p2 = np.linalg.inv(p1)
	p3 = np.dot(p2, X.T)
	w = np.dot(p3, y)
	
	return w
	

def learnRidgeRegression(X,y,lambd):
    #http://statweb.stanford.edu/~owen/courses/305/Rudyregularization.pdf
    # Inputs:
    # X = N x p                                                          
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
	p1 = lambd * X.shape[0]
	Ident = np.identity(X.shape[1])
	p2 = p1 * Ident
	p3 = np.dot(X.T, X)
	p4 = np.add(p2, p3)
	p5 = np.linalg.inv(p4)
	p6 = np.dot(p5, X.T)
	w = np.dot(p6, y)	
	
	return w	

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    # IMPLEMENT THIS METHOD
	N = Xtest.shape[0]
	rmse = 0
	for i in range(N):     
		rmse = rmse + np.square( ytest[i,0]-(np.dot(w.T, Xtest[i,:])) )
	rmse = np.double(np.sqrt(rmse))/N
	return rmse

def regressionObjVal(w, X, y, lambd):

	# compute squared error (scalar) and gradient of squared error with respect
	# to w (vector) for the given data X and y and the regularization parameter
	# lambda                                                                  

	# IMPLEMENT THIS METHOD	
	
	p1 = np.dot(X, w)
	p2 = np.subtract(y, p1)
	p3 = np.dot(p2.T, p2)
	p4 = np.divide(p3, X.shape[0])
	p5 = lambd * w.T
	p6 = np.dot(p5, w)
	
	error1 = np.add(p4, p6)
	error = error1[0, 0]
	
	p7 = ((-2.0) / X.shape[0])
	p8 = np.dot(X.T, y)
	p9 = np.dot(X.T, X)
	p10 = np.dot(p9, w)
	p11 = np.subtract(p8, p10)
	p12 = lambd * X.shape[0]
	p13 = p12 * w
	p14 = np.subtract(p11, p13)
	
	error_grad1 = np.dot(p7, p14)
	error_grad = error_grad1[:, 0]
	
	return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (p+1))                                                         
    # IMPLEMENT THIS METHOD

	Xd = np.ones((x.shape[0], p + 1))
	#print Xd.shape
	#print x.shape
	
	for i in range(0, x.shape[0]):
		for j in range(0, p + 1):
			Xd[i, j] = np.power(x[i], j)
	#Xd = Xd.T
	
	return Xd



######################################### Problem 1 #############################
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('/home/prashanth/Downloads/sample.pickle','rb'))            

# LDA
plt.figure()
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))
plt.show()

##################################### Problem 2 ##################################
X,y,Xtest,ytest = pickle.load(open('/home/prashanth/Downloads/diabetes.pickle','rb'))
  
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle_tr = testOLERegression(w,X,y)
mle_ts = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_tr_i = testOLERegression(w_i,X_i,y)
mle_ts_i = testOLERegression(w_i,Xtest_i,ytest)

print " "
print('RMSE without intercept on train data '+str(mle_tr))
print('RMSE with intercept on train data '+str(mle_tr_i))
print " "
print('RMSE without intercept on test data '+str(mle_ts))
print('RMSE with intercept on test data '+str(mle_ts_i))


################################ Problem 3 ########################################
k = 21
lambdas = np.linspace(0, .004, num=k)
i = 0
rmses3 = np.zeros((k,2))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    #On train data
    rmses3[i,0] = testOLERegression(w_l,X_i,y)
    #On test data
    rmses3[i,1] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.subplot(2, 1, 1)
plt.title('Train data')
plt.plot(lambdas,rmses3[:,0])
plt.subplot(2, 1, 2)
plt.title('Test data')
plt.plot(lambdas,rmses3[:,1])
plt.show()

####################################### Problem 4 ###############################
k = 21
lambdas = np.linspace(0, .004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1

plt.plot(lambdas,rmses4)
plt.plot(lambdas,rmses3)
plt.xlabel('lambda')
plt.title('Figure 1')
plt.axis([0, .5, 4.5, 7])
plt.legend(('RMSE 4','RMSE 3'))
plt.show()

######################################### Problem 5 ###############################
pmax = 7
lambda_opt = lambdas[np.argmin(rmses3)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
	Xd = mapNonLinear(X[:,2],p)
	Xdtest = mapNonLinear(Xtest[:,2],p)
	w_d1 = learnRidgeRegression(Xd,y,0)
	rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
	w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
	rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax), rmses5)
plt.xlabel('p')
plt.title('Figure 2')
plt.legend(('No Regularization','Regularization'))
plt.show()
#########################################   THE END  ############################################################## 
