# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:54:17 2017

@author: MuraliKrishnaB
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:04:05 2017

@author: MuraliKrishnaB
"""

import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords 

news = pd.read_csv('MsiaAccidentCases-train1.csv', delimiter = ',' ,names = ["Cause" ,"Title Case", "Summary Case"] )
#news = pd.read_csv('osha1.csv', delimiter = ',' ,names = ["Cause" ,"Title Case", "Summary Case", "Extras"] )
news.head()

#Summarize the data by the news class
news.groupby('Cause').describe()
print(news.shape)

#Count the length of each document
length=news['Summary Case'].apply(len)
news=news.assign(Length=length)
news.head()

#Plot the distribution of the document length for each category
news.hist(column='Length',by='Cause',bins=50)

plt.show()

WNlemma = nltk.WordNetLemmatizer()
porter = nltk.PorterStemmer()
#X_train = news.iloc[:, [2]]


def pre_process(text):
    tokens = nltk.word_tokenize(text)
    tokens=[WNlemma.lemmatize(t) for t in tokens]
    tokens=[word for word in tokens if word not in set(stopwords.words('english') + ['wa'])]
    text_after_process=" ".join(tokens)
    return(text_after_process)
    
news['Summary Case'] = news['Summary Case'].apply(pre_process)


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(news['Summary Case'], news['Cause'], test_size = 0.33, random_state = 0)

#Create dtm by using word occurence
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer( )
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape

count_vect.get_feature_names()

#Create dtm by using Term Frequency. 
#Divide the number of occurrences of each word in a document 
#by the total number of words in the document: 
#these new features are called tf for Term Frequencies
#If set use_idf=True, which mean create dtm by using tf_idf

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

#Building Modeling by using Naïve Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tf, y_train)

#Prediction on new documents
docs_new = ['victim fallen ', 'worker killed']

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tf_transformer.transform(X_new_counts)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

predicted = clf.predict(X_new_tfidf)


#Build a pipeline: Combine multiple steps into one
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),  
                     ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
                    ])
    
#Use pipeline to train the model
text_clf.fit(X_train,y_train ) 


# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Test model accuracy
import numpy as np
from sklearn import metrics 
predicted = text_clf.predict(X_test)
#np.mean(predicted == y_test) 
print(metrics.confusion_matrix(y_test, predicted))
print(np.mean(predicted == y_test) )