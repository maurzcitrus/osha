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
corpus = []
for i in range(0, 182):
    review = re.sub('[^a-zA-Z]',' ',news['Summary Case'][i])
    review = review.lower()
    #review_pos = pos_tag(word_tokenize(review))
    review = porter.stem(review)
    review_tkn = word_tokenize(review)   
    review_tkn =[WNlemma.lemmatize(t) for t in review_tkn]
    review = [word for word in review_tkn if not word in set(stopwords.words('english') + ['wa'])]
    review = ' '.join(review)
    corpus.append(review)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 810)
X = cv.fit_transform(corpus).toarray()
y = news.iloc[:, 0]

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X = tf_transformer.transform(X).toarray()
X.shape

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Fitting K-NN to the Training set
#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier(n_neighbors = 9, metric = 'minkowski', p = 2)
#classifier.fit(X_train, y_train)

# Fitting Decision Tree Classification to the Training set
#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn import metrics 
cm = confusion_matrix(y_test, y_pred)

print(metrics.confusion_matrix(y_test, y_pred))
print(np.mean(y_pred == y_test) )