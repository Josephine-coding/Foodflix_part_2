from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sentence_transformers import SentenceTransformer
import pandas as pd

##### DATA #####
df = pd.read_csv('/home/apprenant/PycharmProjects/Foodflix_part_2/Data/intermediate.csv', nrows=20000)
cols=['product_name', 'generic_name', 'brands', 'categories']
df=df[cols]
df['content'] = df[['product_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
df['content'].fillna('Null', inplace = True)

def get_results_tfidf(product):
    tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1, 2), min_df = 0, stop_words = 'english')
    tfidf_matrix = tf.fit_transform(df['content'])
    tfidf_product = tf.transform([product])
    cosine_similarities = linear_kernel(tfidf_product, tfidf_matrix).flatten()
    results = cosine_similarities.argsort()[:-6:-1]
    return results

def get_results_count(product):
    vectorizer = CountVectorizer()
    vect_matrix = vectorizer.fit_transform(df['content'])
    vect_product = vectorizer.transform([product])
    cosine_similarities = linear_kernel(vect_product, vect_matrix).flatten()
    results = cosine_similarities.argsort()[:-6:-1]
    return results

def get_results_bert(product):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    sentences = df['content'].head(100).tolist()
    sentences_embeddings = model.encode(sentences)
    sentence_emb=model.encode([product])
    cosine_similarities = linear_kernel(sentence_emb, sentences_embeddings)
    results_sent={}
    similar_indices  =cosine_similarities[0].argsort()[:-6:-1]
    results_sent=[(cosine_similarities[0][i], sentences[i]) for i in similar_indices]
    return results_sent