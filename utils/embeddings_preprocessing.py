import pandas as pd
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors


#clustering sigmalaw bin file with word2vec models
word2vec_model = KeyedVectors.load_word2vec_format("/content/drive/MyDrive/NLP_project/data/legalrawwoldmodel.bin", binary=True, unicode_errors='ignore') 

# cluster the word embeddings into 500 clusters
kmeans = KMeans(n_clusters=500, random_state=42)
clusters = kmeans.fit_predict(word2vec_model.vectors)

# df with words and their clusters
df = pd.DataFrame({
    'Word': word2vec_model.index_to_key,
    'Cluster': clusters
})
# Save as TSV
df.to_csv('/content/drive/MyDrive/NLP_project/data/law_clusters.tsv', sep='\t', index=False)




