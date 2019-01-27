# Working with Text Data

In Representation of Data and features engineering, we talked about two kinds of features that can represent properties of the data: continuous features that describe a quantity, and categorical features that are items from a fixed list. There is a third kind of feature that can be found in many applications, which is text. For example, if we want to classify an email message as either a legitimate email or spam, the content of the email will certainly contain important information for this classification task. Or maybe we want to learn about the opinion of a politician on the topic of immigration. Here, that individual’s speeches or tweets might provide useful information. In customer service, we often want to find out if a message is a complaint or an inquiry. We can use the subject line and content of a message to automatically determine the customer’s intent, which allows us to send the message to the appropriate department, or even send a fully automatic reply.

Text data is usually represented as strings, made up of characters. In any of the examples just given, the length of the text data will vary. This feature is clearly very different from the numeric features that we’ve discussed so far, and we will need to process the data before we can apply our machine learning algorithms to it.

# Types of Data Represented as Strings

Before we dive into the processing steps that go into representing text data for machine learning, we want to briefly discuss different kinds of text data that you might encounter. Text is usually just a string in your dataset, but not all string features should be treated as text. A string feature can sometimes represent categorical variables. There is no way to know how to treat a string feature before looking at the data.


There are four kinds of string data you might see: 
    * Categorical Data
    * Free string that cam be semantically mapped to categories
    * Structured string data
    * Text data
    
Categorical data is data that comes from a fixed list. Say you collect data via a survey where you ask people their favorite color, with a drop-down menu that allows them to select from “red,” “green,” “blue,” “yellow,” “black,” “white,” “purple,” and “pink.” This will result in a dataset with exactly eight different possible values, which clearly encode a categorical variable. You can check whether this is the case for your data by eyeballing it (if you see very many different strings it is unlikely that this is a categorical variable) and confirm it by computing the unique values over the dataset, and possibly a histogram over how often each appears. You also might want to check whether each variable actually corresponds to a category that makes sense for your application. Maybe halfway through the existence of your survey, someone found that “black” was misspelled as “blak” and subsequently fixed the survey. As a result, your dataset contains both “blak” and “black,” which correspond to the same semantic meaning and should be consolidated. 

Now imagine instead of providing a drop-down menu, you provide a text field for the users to provide their own favorite colors. Many people might respond with a color name like “black” or “blue.” Others might make typographical errors, use different spellings like “gray” and “grey,” or use more evocative and specific names like “midnight blue.” You will also have some very strange entries. Some good examples come from the xkcd Color survey(https://blog.xkcd.com/2010/05/03/color-survey-results/), where people had to name colors and came up with names like “velociraptor cloaka” and “my dentist’s office orange. I still remember his dandruff slowly wafting into my gaping yaw,” which are hard to map to colors automatically (or at all). The responses you can obtain from a text field belong to the second category in the list, free strings that can be semantically mapped to categories. It will probably be best to encode this data as a categorical variable, where you can select the categories either by using the most common entries, or by defining categories that will capture responses in a way that makes sense for your application. You might then have some categories for standard colors, maybe a category “multicolored” for people that gave answers like “green and red stripes,” and an “other” category for things that cannot be encoded otherwise. This kind of preprocessing of strings can take a lot of manual effort and is not easily automated. If you are in a position where you can influence data collection, we highly recommend avoiding manually entered values for concepts that are better captured using categorical variables.

Often, manually entered values do not correspond to fixed categories, but still have some underlying structure, like addresses, names of places or people, dates, telephone numbers, or other identifiers. These kinds of strings are often very hard to parse, and their treatment is highly dependent on context and domain.

The final category of string data is freeform text data that consists of phrases or sentences. Examples include tweets, chat logs, and hotel reviews, as well as the collected works of Shakespeare, the content of Wikipedia, or the Project Gutenberg collection of 50,000 ebooks. All of these collections contain information mostly as sentences composed of words.1 For simplicity’s sake, let’s assume all our documents are in one language, English.2 In the context of text analysis, the dataset is often called the corpus, and each data point, represented as a single text, is called a document. These terms come from the information retrieval (IR) and natural language processing (NLP) community, which both deal mostly in text data.

# 1 - Example Application- Sentiment Analysis of Movie Reviews

In Representation of Data and features engineering, we talked about two kinds of features that can represent properties of the data: continuous features that describe a quantity, and categorical features that are items from a fixed list. There is a third kind of feature that can be found in many applications, which is text. For example, if we want to classify an email message as either a legitimate email or spam, the content of the email will certainly contain important information for this classification task. Or maybe we want to learn about the opinion of a politician on the topic of immigration. Here, that individual’s speeches or tweets might provide useful information. In customer service, we often want to find out if a message is a complaint or an inquiry. We can use the subject line and content of a message to automatically determine the customer’s intent, which allows us to send the message to the appropriate department, or even send a fully automatic reply.

Text data is usually represented as strings, made up of characters. In any of the examples just given, the length of the text data will vary. This feature is clearly very different from the numeric features that we’ve discussed so far, and we will need to process the data before we can apply our machine learning algorithms to it.

# Types of Data Represented as Strings

Before we dive into the processing steps that go into representing text data for machine learning, we want to briefly discuss different kinds of text data that you might encounter. Text is usually just a string in your dataset, but not all string features should be treated as text. A string feature can sometimes represent categorical variables. There is no way to know how to treat a string feature before looking at the data.


There are four kinds of string data you might see: 
    * Categorical Data
    * Free string that cam be semantically mapped to categories
    * Structured string data
    * Text data
    
Categorical data is data that comes from a fixed list. Say you collect data via a survey where you ask people their favorite color, with a drop-down menu that allows them to select from “red,” “green,” “blue,” “yellow,” “black,” “white,” “purple,” and “pink.” This will result in a dataset with exactly eight different possible values, which clearly encode a categorical variable. You can check whether this is the case for your data by eyeballing it (if you see very many different strings it is unlikely that this is a categorical variable) and confirm it by computing the unique values over the dataset, and possibly a histogram over how often each appears. You also might want to check whether each variable actually corresponds to a category that makes sense for your application. Maybe halfway through the existence of your survey, someone found that “black” was misspelled as “blak” and subsequently fixed the survey. As a result, your dataset contains both “blak” and “black,” which correspond to the same semantic meaning and should be consolidated. 

# Working with the IMBd Dataset

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/7%20-%20Working%20with%20Text%20Data/images/img1.PNG)

# Bag of Words representation

One of the most simple but effective and commonly used ways to represent text for machine learning is using the bag-of-words representation. When using this representation, we discard most of the structure of the input text, like chapters, paragraphs, sentences, and formatting, and only count how often each word appears in each text in the corpus. Discarding the structure and counting only word occurrences leads to the mental image of representing text as a “bag.”

Computing the bag-of-words representation for a corpus of documents consists of the following three steps: 

1 - Tokenization. Split each document into the words that appear in it (called tokens), for example by splitting them on whitespace and punctuation.

2 - Vocabulary building. Collect a vocabulary of all words that appear in any of the documents, and number them (say, in alphabetical order).

3 - Encoding. For each document, count how often each of the words in the vocabulary appear in this document. 

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/7%20-%20Working%20with%20Text%20Data/images/img2.PNG)


# Rescaling the Data with tf–idf

Instead of dropping features that are deemed unimportant, another approach is to rescale features by how informative we expect them to be. One of the most common ways to do this is using the term frequency–inverse document frequency (tf–idf) method. The intuition of this method is to give high weight to any term that appears often in a particular document, but not in many documents in the corpus. If a word appears often in a particular document, but not in very many documents, it is likely to be very descriptive of the content of that document. scikit-learn implements the tf–idf method in two classes: TfidfTransformer, which takes in the sparse matrix output produced by CountVectorizer and transforms it, and TfidfVectorizer, which takes in the text data and does both the bag-of-words feature extraction and the tf–idf transformation.  The tf–idf score for word w in document d as implemented in both the TfidfTransformer and TfidfVectorizer classes is given by:

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/7%20-%20Working%20with%20Text%20Data/images/img3.PNG)

where N is the number of documents in the training set, Nw is the number of documents in the training set that the word w appears in, and tf (the term frequency) is the number of times that the word w appears in the query document d (the document you want to transform or encode). Both classes also apply L2 normalization after computing the tf–idf representation; in other words, they rescale the representation of each document to have Euclidean norm 1. Rescaling in this way means that the length of a document (the number of words) does not change the vectorized representation.

Because tf–idf actually makes use of the statistical properties of the training data, we will use a pipeline, to ensure the results of our grid search are valid.


# Bag-of-Words with More Than One Word (n-Grams)

One of the main disadvantages of using a bag-of-words representation is that word order is completely discarded. Therefore, the two strings “it’s bad, not good at all” and “it’s good, not bad at all” have exactly the same representation, even though the meanings are inverted. Putting “not” in front of a word is only one example (if an extreme one) of how context matters. Fortunately, there is a way of capturing context when using a bag-of-words representation, by not only considering the counts of single tokens, but also the counts of pairs or triplets of tokens that appear next to each other. Pairs of tokens are known as bigrams, triplets of tokens are known as trigrams, and more generally sequences of tokens are known as n-grams. We can change the range of tokens that are considered as features by changing the ngram_range parameter of CountVectorizer or TfidfVectorizer. The ngram_range parameter is a tuple, consisting of the minimum length and the maximum length of the sequences of tokens that are considered.

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/7%20-%20Working%20with%20Text%20Data/images/img4.PNG)
   * 1 gram word
   
![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/7%20-%20Working%20with%20Text%20Data/images/img5.PNG)
   * 3-gram words

(To understand how bag of words actually works, see above code 1 - Sentiment Analysis of Movie Reviews)


# 3 - Advanced Tokenization, Stemming, and Lemmatization
   (No description here, Everything is explained into code. See above file "3 - Advanced Tokenization, Stemming, and Lemmatization3 - Advanced Tokenization, Stemming, and Lemmatization")
   
# 4 - Topic Modeling and Document Clustering

One particular technique that is often applied to text data is topic modeling, which is an umbrella term describing the task of assigning each document to one or multiple topics, usually without supervision. A good example for this is news data, which might be categorized into topics like “politics,” “sports,” “finance,” and so on. If each document is assigned a single topic, this is the task of clustering the documents, as discussed in Unsupervised Learning.If each document can have more than one topic, the task relates to the decomposition methods from Unsupervised Learning. Each of the components we learn then corresponds to one topic, and the coefficients of the components in the representation of a document tell us how strongly related that document is to a particular topic. Often, when people talk about topic modeling, they refer to one particular decomposition method called Latent Dirichlet Allocation (often LDA for short).

# 4.1 Latent Dirichlet Allocation

Intuitively, the LDA model tries to find groups of words (the topics) that appear together frequently. LDA also requires that each document can be understood as a “mixture” of a subset of the topics. It is important to understand that for the machine learning model a “topic” might not be what we would normally call a topic in everyday speech, but that it resembles more the components extracted by PCA or NMF, which might or might not have a semantic meaning. Even if there is a semantic meaning for an LDA “topic”, it might not be something we’d usually call a topic. Going back to the example of news articles, we might have a collection of articles about sports, politics, and finance, written by two specific authors. In a politics article, we might expect to see words like “governor,” “vote,” “party,” etc., while in a sports article we might expect words like “team,” “score,” and “season.” Words in each of these groups will likely appear together, while it’s less likely that, for example, “team” and “governor” will appear together. However, these are not the only groups of words we might expect to appear together. The two reporters might prefer different phrases or different choices of words. Maybe one of them likes to use the word “demarcate” and one likes the word “polarize.” Other “topics” would then be “words often used by reporter A” and “words often used by reporter B,” though these are not topics in the usual sense of the word.

(For more information see the code in filename "4 - Topic Modeling and Document Clustering". Well explained with code.)

![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/7%20-%20Working%20with%20Text%20Data/images/img6.PNG)

* Way to inspect the topics weights
![alt text](https://github.com/manish29071998/Introduction-to-Machine-Learning-with-Python/blob/master/7%20-%20Working%20with%20Text%20Data/images/img7.PNG)
