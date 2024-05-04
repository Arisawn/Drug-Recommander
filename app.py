import streamlit as st
import pickle
import pandas as pd

df=pd.read_csv('drugsComTrain.csv')

nan_indices = df[df['condition'].isnull()].index
df = df.drop(nan_indices)

import joblib
vectorizer = joblib.load('tfidfvectorizer.pkl')
model = joblib.load('passmodel.pkl')

def top_drugs_extractor(condition):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst

st.title('Medicine Recommender System')


sentence = st.text_input("Enter your symptoms:")


                                 
if st.button('Recommend Medicine'):
     test = model.predict(vectorizer.transform([sentence]))
     st.write(test[0])
     st.write(top_drugs_extractor(test[0]))