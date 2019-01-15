# 5 - Metrics for Binary Classification

Binary classification is arguably the most common and conceptually simple application of machine learning in practice. However, there are still a number of caveats in evaluating even this simple task. Before we dive into alternative metrics, let’s have a look at the ways in which measuring accuracy might be misleading. Remember that for binary classification, we often speak of a positive class and a negative class, with the understanding that the positive class is the one we are looking for.

# 5.1 - Kinds of error

Often, accuracy is not a good measure of predictive performance, as the number of mistakes we make does not contain all the information we are interested in. Imagine an application to screen for the early detection of cancer using an automated test. If the test is negative, the patient will be assumed healthy, while if the test is positive, the patient will undergo additional screening. Here, we would call a positive test (an indication of cancer) the positive class, and a negative test the negative class. We can’t assume that our model will always work perfectly, and it will make mistakes. For any application, we need to ask ourselves what the consequences of these mistakes
might be in the real world. 

One possible mistake is that a healthy patient will be classified as positive, leading to additional testing. This leads to some costs and an inconvenience for the patient (and possibly some mental distress). An incorrect positive prediction is called a false positive. The other possible mistake is that a sick patient will be classified as negative, and will not receive further tests and treatment. The undiagnosed cancer might lead to serious health issues, and could even be fatal. A mistake of this kind—an incorrect negative prediction—is called a false negative. In statistics, a false positive is also known as type I error, and a false negative as type II error. We will stick to “false negative” and “false positive,” as they are more explicit and easier to remember. In the cancer diagnosis example, it is clear that we want to avoid false negatives as much as possible, while false positives can be viewed as more of a minor nuisance. 

While this is a particularly drastic example, the consequence of false positives and false negatives are rarely the same. In commercial applications, it might be possible to assign dollar values to both kinds of mistakes, which would allow measuring the error of a particular prediction in dollars, instead of accuracy. This might be much more meaningful for making business decisions on which model to use.


# 5.2-  Imbalanced Dataset

Types of errors play an important role when one of two classes is much more frequent than the other one. This is very common in practice; a good example is click-through prediction, where each data point represents an “impression,” an item that was shown to a user. This item might be an ad, or a related story, or a related person to follow on a social media site. The goal is to predict whether, if shown a particular item, a user will click on it (indicating they are interested). Most things users are shown on the Internet (in particular, ads) will not result in a click. You might need to show a user 100 ads or articles before they find something interesting enough to click on. This results in a dataset where for each 99 “no click” data points, there is 1 “clicked” data point; in other words, 99% of the samples belong to the “no click” class. Datasets in which one class is much more frequent than the other are often called imbalanced datasets, or datasets with imbalanced classes. In reality, imbalanced data is the norm, and it is rare that the events of interest have equal or even similar frequency in the data. 


# 5.3 Confusion Matrics

One of the most comprehensive ways to represent the result of evaluating binary classification is using confusion matrices. Let’s inspect the predictions of LogisticRegres sion from the previous section using the confusion_matrix function. We already stored the predictions on the test set in pred_logreg:

The output of confusion_matrix is a two-by-two array, where the rows correspond to the true classes and the columns correspond to the predicted classes. Each entry counts how often a sample that belongs to the class corresponding to the row (here, “not nine” and “nine”) was classified as the class corresponding to the column. The following plot illustrates this meaning:

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/5%20-%20Model%20Evaluation%20and%20Improvement/images/img1.PNG)

Entries on the main diagonal3 of the confusion matrix correspond to correct classifications, while other entries tell us how many samples of one class got mistakenly classified as another class. 

If we declare “a nine” the positive class, we can relate the entries of the confusion matrix with the terms false positive and false negative that we introduced earlier. To complete the picture, we call correctly classified samples belonging to the positive class true positives and correctly classified samples belonging to the negative class true negatives. These terms are usually abbreviated FP, FN, TP, and TN and lead to the following interpretation for the confusion matrix 

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/5%20-%20Model%20Evaluation%20and%20Improvement/images/img2.PNG)

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/5%20-%20Model%20Evaluation%20and%20Improvement/images/img3.PNG)

Looking at the confusion matrix, it is quite clear that something is wrong with pred_most_frequent, because it always predicts the same class. pred_dummy, on the other hand, has a very small number of true positives (4), particularly compared to the number of false negatives and false positives—there are many more false positives than true positives! The predictions made by the decision tree make much more sense than the dummy predictions, even though the accuracy was nearly the same. Finally, we can see that logistic regression does better than pred_tree in all aspects: it has more true positives and true negatives while having fewer false positives and false negatives. From this comparison, it is clear that only the decision tree and the logistic regression give reasonable results, and that the logistic regression works better than the tree on all accounts. However, inspecting the full confusion matrix is a bit cumbersome, and while we gained a lot of insight from looking at all aspects of the matrix, the process was very manual and qualitative

# Relation to Accuracy
 We already saw one way to summarize the result in the confusion matrix—by computing accuracy, which can be expressed as: 
 
 ![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/5%20-%20Model%20Evaluation%20and%20Improvement/images/img4.PNG)
 
 In other words, accuracy is the number of correct predictions (TP and TN) divided by the number of all samples (all entries of the confusion matrix summed up). 
 
 # Precision, recall and f-score: 
 There are several other ways to summarize the confusion matrix, with the most common ones being precision and recall. Precision measures how many of the samples predicted as positive are actually positive:

 ![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/5%20-%20Model%20Evaluation%20and%20Improvement/images/img5.PNG)
 
 Precision is used as a performance metric when the goal is to limit the number of false positives. As an example, imagine a model for predicting whether a new drug will be effective in treating a disease in clinical trials. Clinical trials are notoriously expensive, and a pharmaceutical company will only want to run an experiment if it is very sure that the drug will actually work. Therefore, it is important that the model does not produce many false positives—in other words, that it has a high precision. Precision is also known as positive predictive value (PPV). 
 
 # Recall:
 On the other hand, measures how many of the positive samples are captured by the positive predictions: 
 
 ![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/5%20-%20Model%20Evaluation%20and%20Improvement/images/img6.PNG)
 
 Recall is used as performance metric when we need to identify all positive samples; that is, when it is important to avoid false negatives. The cancer diagnosis example from earlier in this chapter is a good example for this: it is important to find all people that are sick, possibly including healthy patients in the prediction. Other names for recall are sensitivity, hit rate, or true positive rate (TPR). 
 
 There is a trade-off between optimizing recall and optimizing precision. You can trivially obtain a perfect recall if you predict all samples to belong to the positive class— there will be no false negatives, and no true negatives either. However, predicting all samples as positive will result in many false positives, and therefore the precision will be very low. On the other hand, if you find a model that predicts only the single data point it is most sure about as positive and the rest as negative, then precision will be perfect (assuming this data point is in fact positive), but recall will be very bad.


So, while precision and recall are very important measures, looking at only one of them will not provide you with the full picture. One way to summarize them is the f-score or f-measure, which is with the harmonic mean of precision and recall: 


 ![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/5%20-%20Model%20Evaluation%20and%20Improvement/images/img7.PNG)
 
 # Taking uncertainty into account:
 
 ![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/5%20-%20Model%20Evaluation%20and%20Improvement/images/img8.PNG) 
   Heatmap of the decision function and the impact of changing the decision threshold

To know more, explore the code given above...
