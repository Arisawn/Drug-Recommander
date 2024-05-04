import pandas as pd 
# import itertools 
import string
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from bs4 import BeautifulSoup
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

df=pd.read_csv('drugsComTrain.csv')

nan_indices = df[df['condition'].isnull()].index
df = df.drop(nan_indices)

# df_sorted = df.sort_values(by=['condition', 'rating'], ascending=[True, False])
# def truncate_group(group):
#     return group.head(200)
# grouped = df_sorted.groupby('condition', group_keys=False)
# df_truncated = grouped.apply(truncate_group)
# df_truncated = df_truncated.reset_index(drop=True)
# print(df_truncated)

# Y_var = df['condition']
# X_var = df['review']

# x_train, x_test, Y_train, Y_test = train_test_split(X_var, Y_var) 

# tfidf_vectorizer3 = TfidfVectorizer(stop_words='english', max_df=0.8, ngram_range=(1,3))
# tfidf_train_3 = tfidf_vectorizer3.fit_transform(x_train)
# tfidf_test_3 = tfidf_vectorizer3.transform(x_test)

# pass_tf = PassiveAggressiveClassifier()
# pass_tf.fit(tfidf_train_3, Y_train)
# pred = pass_tf.predict(tfidf_test_3)
# score = metrics.accuracy_score(Y_test, pred)
# print("accuracy:   %0.3f" % score)


def top_drugs_extractor(condition):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst

# lemmatizer = WordNetLemmatizer()

# stop = stopwords.words('english')

# def review_to_words(raw_review):
#     review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
#     letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
#     words = letters_only.lower().split()
#     meaningful_words = [w for w in words if not w in stop]
#     lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
#     return( ' '.join(lemmitize_words))

# def predict_text(lst_text):
#     df_test = pd.DataFrame(lst_text, columns = ['test_sent'])
#     df_test["test_sent"] = df_test["test_sent"].apply(review_to_words)
#     tfidf_bigram = tfidf_vectorizer3.transform(lst_text)
#     prediction = pass_tf.predict(tfidf_bigram)
#     df_test['prediction']=prediction
#     return df_test

import joblib
# joblib.dump(tfidf_vectorizer3, 'tfidfvectorizer.pkl', compress=True)
# joblib.dump(pass_tf, 'passmodel.pkl', compress=True)

vectorizer = joblib.load('tfidfvectorizer.pkl')
model = joblib.load('passmodel.pkl')

sentence = "i had ice cream and ice cream was very cold and i had hot  chocolate and now i have fever and also some cough but now i am feeling sleepy"

test = model.predict(vectorizer.transform([sentence]))
print(test[0])
print(top_drugs_extractor(test[0]))