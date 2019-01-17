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


# 2. Convenient Pipeline Creation

Creating a pipeline using the syntax described earlier is sometimes a bit cumbersome, and we often don’t need user-specified names for each step. There is a convenience function, make_pipeline, that will create a pipeline for us and automatically name each step based on its class.
  Tutorials are expained better in "2 - The General Pipeline Interface" file.
  
  ![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/6%20-%20Algorithms%20chains%20and%20Pipelines/images/img4.PNG)
  
  
 # 3 - Grid-Search Preprocessing Steps and Model Parameters
 
 Using pipelines, we can encapsulate all the processing steps in our machine learning workflow in a single scikit-learn estimator. Another benefit of doing this is that we can now adjust the parameters of the preprocessing using the outcome of a supervised task like regression or classification.
 
 ![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/6%20-%20Algorithms%20chains%20and%20Pipelines/images/img3.PNG)
 
 # 3.1 - Grid-Searching which model to use
 
 In the file "3 - Grid-Searching Preprocessing Steps and Model Parameter"
 Believe me, it's awesome. You should read.
