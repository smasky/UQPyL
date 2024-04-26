'''
    Examples for using surrogate models:
    - 1. full_connect_neural_network (FNN)
    - 2. gaussian_process (GP) 
    - 3. kriging (KRG)
    - 4. linear regression (LR)
    - 5. polynomial regression (PR)
    - 6. radial basis function (RBF)
    - 7. support vector machine (SVM)
    - 8. multivariate adaptive regression splines (MARS)
'''
#temple
import sys
sys.path.append(".")
from scipy.io import loadmat
import os
os.chdir('./examples')

#Evaluation Metrics
from UQPyL.utility.metrics import r2_score
from UQPyL.utility.metrics import rank_score

#Scaling for X and Y
from UQPyL.utility.scalers import MinMaxScaler

#############Prepare Data#################
#Use Sphere function
from UQPyL.problems import Sphere
#F= \sum x_i
# we use 10 variables here with 300 samples
problem = Sphere(n_input=10)
#DOE: LHS is used
from UQPyL.DoE import LHS
lhs=LHS('classic')
#train set
X_train=lhs.sample(300, 10)
Y_train=problem.evaluate(X_train, unit=True)
#test set
X_test=lhs.sample(50, 10)
Y_test=problem.evaluate(X_test, unit=True)
###########################################

############4. linear regression (LR)#######
print('############4. linear regression (LR)#######')
from UQPyL.surrogates.linear_regression import LinearRegression
#There three types: 'Origin', 'Lasso', 'Ridge'
# Main difference is the loss function
# Origin
lr=LinearRegression(loss_type='Origin')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_test)
#use R-square to validate the Y_predict and Y_test
r2=r2_score(Y_test, Y_predict)
print(r2)
rank=rank_score(Y_test, Y_predict)
print(rank)
# Ridge
lr=LinearRegression(loss_type='Ridge')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_test)
#use R-square to validate the Y_predict and Y_test
r2=r2_score(Y_test, Y_predict)
print(r2)
rank=rank_score(Y_test, Y_predict)
print(rank)
#Lasso
lr=LinearRegression(loss_type='Lasso')
lr.fit(X_train, Y_train)
Y_predict=lr.predict(X_test)
#use R-square to validate the Y_predict and Y_test
r2=r2_score(Y_test, Y_predict)
print(r2)
rank=rank_score(Y_test, Y_predict)
print(rank)

############5. polynomial regression (PR)####
print('############5. polynomial regression (PR)####')
from UQPyL.surrogates.polynomial_regression import PolynomialRegression
#degree=2 default. degree=1 is linear regression
# there are three types: 'Origin', 'Lasso', 'Ridge'
pr=PolynomialRegression(degree=2, loss_type='Lasso')
pr.fit(X_train, Y_train)
Y_predict=pr.predict(X_test)
#use R-square to validate the Y_predict and Y_test
r2=r2_score(Y_test, Y_predict)
print(r2)
rank=rank_score(Y_test, Y_predict)
print(rank)
# there are three types: 'Origin', 'Lasso', 'Ridge'
pr=PolynomialRegression(degree=2, loss_type='Ridge')
pr.fit(X_train, Y_train)
Y_predict=pr.predict(X_test)
#use R-square to validate the Y_predict and Y_test
r2=r2_score(Y_test, Y_predict)
print(r2)
rank=rank_score(Y_test, Y_predict)
print(rank)

############3. kriging (KRG)#################
print("#################kriging (KRG)#################")
from UQPyL.surrogates.kriging import KRG
from UQPyL.surrogates.krg_kernels import Guass_Kernel

#use guassian kernel, there two fit mode: 'predictError' and 'likelihood'
#use predictError here
guass=Guass_Kernel(theta=1e-3, theta_lb=1e-5, theta_ub=1, heterogeneous=True)
krg=KRG(kernel=guass, optimizer='Boxmin', fitMode='predictError')
krg.fit(X_train,Y_train)
Y_predict=krg.predict(X_test)
#use R-square to validate the Y_predict and Y_test
r2=r2_score(Y_test, Y_predict)
print(r2)
rank=rank_score(Y_test, Y_predict)
print(rank)
#use likehood here
guass=Guass_Kernel(theta=1e-3, theta_lb=1e-5, theta_ub=1, heterogeneous=True)
krg=KRG(kernel=guass, optimizer='GA', fitMode='likelihood')
krg.fit(X_train,Y_train)
Y_predict=krg.predict(X_test)
#use R-square to validate the Y_predict and Y_test
r2=r2_score(Y_test, Y_predict)
print(r2)
rank=rank_score(Y_test, Y_predict)
print(rank)

############2. gaussian_process (GP)#################
print("#################gaussian_process (GP)#################")
from UQPyL.surrogates.gaussian_process import GPR
# we should construct a kernel for GPR
from UQPyL.surrogates.gp_kernels import RBF
rbf_kernel=RBF()
gp=GPR(kernel=rbf_kernel,scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)))
gp.fit(X_train,Y_train)
Y_predict=gp.predict(X_test)
#use R-square to validate the Y_predict and Y_test
r2=r2_score(Y_test, Y_predict)
print(r2)
rank=rank_score(Y_test, Y_predict)
print(rank)


#############1. full_connect_neural_network (FNN)#################
print("#################full_connect_neural_network (FNN)#################")
from UQPyL.surrogates import FNN
#use 0-1 MinMaxScaler to normalize data
fnn=FNN(scalers=(MinMaxScaler(0,1), MinMaxScaler(0,1)),hidden_layer_sizes=[16,32,64,32,16,8], 
        activation_functions='relu', solver='adam', alpha=0.001)
fnn.fit(X_train,Y_train)
Y_predict=fnn.predict(X_test)
#use R-square to validate the Y_predict and Y_test

r2=r2_score(Y_test, Y_predict)
print('r2:', r2)
#use rank_score to validate the Y_predict and Y_test

oe=rank_score(Y_test, Y_predict)
print('rank_score:', oe)

