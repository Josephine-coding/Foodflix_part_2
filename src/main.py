from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

##### DATA #####
df = pd.read_csv('/home/apprenant/PycharmProjects/Foodflix_part_2/Data/intermediate.csv', nrows=20000)
cols=['product_name', 'generic_name', 'brands', 'categories']
df=df[cols]
df['content'] = df[['product_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
df['content'].fillna('Null', inplace = True)

##### TEST BERT #####
word_embedding_model = models.Transformer('camembert-base')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

sentence = ['Chocolat']
sentences = df['content'].head(20).tolist()

sentences_embeddings = model.encode(sentences)
sentence_emb=model.encode(sentence)

cosine_similarities = linear_kernel(sentence_emb, sentences_embeddings)
results_sent={}
similar_indices  =cosine_similarities[0].argsort()[:-5:-1]
results_sent=[(cosine_similarities[0][i], sentences[i]) for i in similar_indices]
print(results_sent)
