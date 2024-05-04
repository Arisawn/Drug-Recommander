import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier,LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv('drugsComTrain.csv')
nan_indices = df[df['condition'].isnull()].index
df = df.drop(nan_indices)
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(df['review'])
X = df['review']
y = df['condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#a)PAC
# passive = PassiveAggressiveClassifier()
# passive.fit(X_train, y_train)
# pred = passive.predict(X_test)
# score = metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)
# 55%


# b) MNB
# clf = MultinomialNB()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# score = metrics.accuracy_score(y_test, y_pred)
# print("accuracy:   %0.3f" % score)
# 51%

#c) using Tfidf output as MNB input
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,3))
# tfidf_train_2 = tfidf_vectorizer.fit_transform(X_train)
# tfidf_test_2 = tfidf_vectorizer.transform(X_test)

# mnb_tf = MultinomialNB()
# mnb_tf.fit(tfidf_train_2, y_train)
# pred = mnb_tf.predict(tfidf_test_2)
# score = metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)
# memory error

#d) using Tfidf output as PAC input
# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8)
# tfidf_train = tfidf_vectorizer.fit_transform(X_train)
# tfidf_test = tfidf_vectorizer.transform(X_test)
# pass_tf = PassiveAggressiveClassifier()
# pass_tf.fit(tfidf_train, y_train)
# pred = pass_tf.predict(tfidf_test)
# score = metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)
# 75.3%

# tfidf_vectorizer2 = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,2))
# tfidf_train = tfidf_vectorizer2.fit_transform(X_train)
# tfidf_test = tfidf_vectorizer2.transform(X_test)
# pass_tf = PassiveAggressiveClassifier()
# pass_tf.fit(tfidf_train, y_train)
# pred = pass_tf.predict(tfidf_test)
# score = metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)
# 80%

# tfidf_vectorizer3 = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,3))
# tfidf_train_3 = tfidf_vectorizer3.fit_transform(X_train)
# tfidf_test_3 = tfidf_vectorizer3.transform(X_test)
# pass_tf = PassiveAggressiveClassifier()
# pass_tf.fit(tfidf_train_3, y_train)
# pred = pass_tf.predict(tfidf_test_3)
# score = metrics.accuracy_score(y_test, pred)
# print("accuracy:   %0.3f" % score)
# 80%