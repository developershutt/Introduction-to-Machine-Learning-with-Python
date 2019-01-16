# Algorithms Chains and Pipelines

# 1 - Introduction

For many machine learning algorithms, the particular representation of the data that you provide is very important. This starts with scaling the data and combining features by hand and goes all the way to learning features using unsupervised machine learning. Consequently, most machine learning applications require not only the application of a single algorithm, but the chaining together of many different processing steps and machine learning models. In this chapter, we will cover how to use the Pipeline class to simplify the process of building chains of transformations and models. In particular, we will see how we can combine Pipeline and GridSearchCV to search over parameters for all processing steps at once.

# 1.1 - Parameters selection and Preprocessing

Explained in code (file is above "1- Introduction.ipynb")

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/6%20-%20Algorithms%20chains%20and%20Pipelines/images/img1.PNG)
  Improper processing
  
  # 1.2 - Using Pipelines in Grid Searchs
  
  Using a pipeline in a grid search works the same way as using any other estimator. We define a parameter grid to search over, and construct a GridSearchCV from the pipeline and the parameter grid. When specifying the parameter grid, there is a slight change, though. We need to specify for each parameter which step of the pipeline it belongs to.
  
  ![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/6%20-%20Algorithms%20chains%20and%20Pipelines/images/img2.PNG)
    Proper processing
