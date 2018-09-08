# 1. Introduction-to-Machine-Learning-with-Python
This course is for absolute beginners who have elementary knowledge of Python Programming Language with scientific libs like Numpy, matplotlib and pandas. So lets get started....

* Know Your Data
* Look at your data
![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/images/iris_data_visualization.PNG?raw=true)
* Split the Dataset for both Training and Testing
* Build Classifier and fit the training data
* Measure Success of Test data
* Prediction
* Evaluating your model


# 2.Supervised Machine Learning
* Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs.

* Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output.

# 3. KNeighbors Regressor with wave dataset
In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. 
![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/images/knn.PNG)

# 4. Linear Models of Classification

# 4.1 Linear Regression
Linear regression, or ordinary least squares (OLS), is the simplest and most classic linear method for regression. Linear regression finds the parameters w and b that minimize the mean squared error between predictions and the true regression targets, y, on the training set. The mean squared error is the sum of the squared differences between the predictions and the true values. Linear regression has no parameters, which is a benefit, but it also has no way to control model complexity. By using wave dataset :
![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/images/linear_regression.PNG)

# 4.2 Ridge Regression
Ridge regression is also a linear model for regression, so the formula it uses to make predictions is the same one used for ordinary least squares. In ridge regression, though, the coefficients (w) are chosen not only so that they predict well on the training data, but also to fit an additional constraint. We also want the magnitude of coefficients to be as small as possible; in other words, all entries of w should be close to zero. Intuitively, this means each feature should have as little effect on the outcome as possible (which translates to having a small slope), while still predicting well. This constraint is an example of what is called regularization. Regularization means explicitly restricting a model to avoid overfitting. The particular kind used by ridge regression is known as L2 regularization.
![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/images/ridge_regression.PNG)

# Linear Regression VS Ridge Regression
![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/images/LR_vs_RR.PNG)

# 4.3 Lasso
An alternative to Ridge for regularization linear regression is lasso. As with ridge regression, using lasso also restricts coeficients to be close to zero but in a slightly different way, called L1 regularization.8 The consequence of L1 regularization is that when using the lasso, some coefficients are exactly zero. This means some features are entirely ignored by the model. This can be seen as a form of automatic feature selection. Having some coefficients be exactly zero often makes a model easier to interpret, and can reveal the most important features of your model.

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/images/lasso.PNG)
