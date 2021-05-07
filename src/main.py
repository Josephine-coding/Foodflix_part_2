import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.write("""
# Foodflix
""")
st.subheader("*Recommandation de produits*")

st.sidebar.header("About")
st.sidebar.text("Test")

product = st.text_input("Quel produit cherchez vous?", " ")

st.write("Vous avez choisi ce produit : ",product)


df = pd.read_csv('/home/apprenant/PycharmProjects/Foodflix_part_2/Data/intermediate.csv', nrows=20000)
cols=['product_name', 'generic_name', 'brands', 'categories']
df=df[cols]

df['content'] = df[['product_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
df['content'].fillna('Null', inplace = True)

tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(df['content'])
tfidf_product = tf.transform([product])
cosine_similarities = linear_kernel(tfidf_product, tfidf_matrix).flatten()
results = cosine_similarities.argsort()[:-5:-1]


for i in range(4):
    st.write(df.iloc[results[i]])


