import streamlit as st
import pandas as pd

import sys
sys.path.insert(0, "/home/apprenant/PycharmProjects/Foodflix_part_2")
from utils import get_results_tfidf, get_results_count, get_results_bert

##### DATA #####
df = pd.read_csv('/home/apprenant/PycharmProjects/Foodflix_part_2/Data/intermediate.csv', nrows=20000)
cols=['product_name', 'generic_name', 'brands', 'categories']
#df=df[cols]
df['content'] = df[['product_name','generic_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
df['content'].fillna('Null', inplace = True)

##### SIDEBAR #####
st.sidebar.write("# Foodflix")
st.sidebar.text("Choissisez le modèle de vectorisation")
model = st.sidebar.radio('', ["TFIDF","Count Vectorizer","BERT"])
st.sidebar.write("# Quel produit cherchez vous?")
product = st.sidebar.text_input(" ", " ")

##### BODY #####
st.write("# Recommandation de produits")

if model == "TFIDF":
    results = get_results_tfidf(product)
    for i in range(5):
        st.write("* ", df['product_name'].iloc[results[i]])
        my_expander = st.beta_expander("Plus d'infos sur le produit", expanded=False)
        with my_expander:
            st.header(df['product_name'].iloc[results[i]])
            st.subheader(f"Marque : {df['brands'].iloc[results[i]]}")
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown(f"**_Valeurs énergétiques (pour 100g): {df['energy_100g'].iloc[results[i]]}_**")
                st.text("Categorie du produit :")
                st.write(df['categories'].iloc[results[i]])
            with col2:
                st.info(f"**_Nutri-Score: {df['nutrition_grade_fr'].iloc[results[i]]}_**")
            st.markdown("_______")

elif model == "Count Vectorizer":
    results = get_results_count(product)
    for i in range(5):
        st.write("* ", df['product_name'].iloc[results[i]])
        my_expander = st.beta_expander("Plus d'infos sur le produit", expanded=False)
        with my_expander:
            st.header(df['product_name'].iloc[results[i]])
            st.subheader(f"Marque : {df['brands'].iloc[results[i]]}")
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown(f"**_Valeurs énergétiques (pour 100g): {df['energy_100g'].iloc[results[i]]}_**")
                st.text("Categorie du produit :")
                st.write(df['categories'].iloc[results[i]])
            with col2:
                st.info(f"**_Nutri-Score: {df['nutrition_grade_fr'].iloc[results[i]]}_**")
            st.markdown("_______")

elif model == "BERT":
    results = get_results_bert(product)
    st.warning('Cette page est en cours de construction')
    st.dataframe(results)