from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import pandas as pd

test_csv = pd.read_csv('test.csv',names=['x','y']) # path to file
train_csv = pd.read_csv('large.csv',names=['x','y']) # path to file

train_X = train_csv['x']   # '0' corresponds to Texts/Reviews
train_y = train_csv['y']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X = test_csv['x']
test_y = test_csv['y']

t = time()  # not compulsory

# loading CountVectorizer
tf_vectorizer = CountVectorizer() # or term frequency
tfidf_vectorizer=TfidfVectorizer()

X_train_tf = tf_vectorizer.fit_transform(train_X)
X_train_tfidf=tfidf_vectorizer.fit_transform(train_X)

duration = time() - t
print("Time taken to extract features from training data : %f seconds" % (duration))
print("n_samples: %d, n_features: %d" % X_train_tf.shape)

t = time()
X_test_tf = tf_vectorizer.transform(test_X)
X_test_tfidf=tfidf_vectorizer.transform(test_X)

duration = time() - t
print("Time taken to extract features from test data : %f seconds" % (duration))
print("n_samples: %d, n_features: %d" % X_test_tf.shape)

# build naive bayes classification model
t = time()

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)

training_time = time() - t
print("train time: %0.3fs" % training_time)
# predict the new document from the testing dataset
t = time()
y_pred = naive_bayes_classifier.predict(X_test_tfidf)

test_time = time() - t
print("test time:  %0.3fs" % test_time)

# compute the performance measures
score1 = metrics.accuracy_score(test_y, y_pred)
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(test_y, y_pred,
                                            target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(test_y, y_pred))

print('------------------------------')
