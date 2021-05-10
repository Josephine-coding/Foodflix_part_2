from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

##### DATA #####
df = pd.read_csv('/home/apprenant/PycharmProjects/Foodflix_part_2/Data/intermediate.csv', nrows=20000)
cols=['product_name', 'generic_name', 'brands', 'categories']
df=df[cols]
df['content'] = df[['product_name', 'categories']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
df['content'].fillna('Null', inplace = True)

##### EXAMPLE SHOWING HOW SENTENCE TRANSFORMER WORKS  #####

model = SentenceTransformer('paraphrase-distilroberta-base-v1')

sentence = ['This framework generates embeddings for each input sentence']
sentences = ['Sentences are passed as a list of strings',
             'This framework generates embeddings for each sentence',
             'This framework generates things',
             'The quick brown fox jumps over the lazy dog']

# sentence = ['chocolat']
# sentences = df['content'].head(50).tolist()

sentences_embeddings = model.encode(sentences)
sentence_emb=model.encode(sentence)

cosine_similarities = linear_kernel(sentence_emb, sentences_embeddings)
results_sent={}
similar_indices  =cosine_similarities[0].argsort()[:-5:-1]
results_sent=[(cosine_similarities[0][i], sentences[i]) for i in similar_indices]

print("La phrase la plus ressemblante est : ", results_sent[0][1])
print(results_sent)