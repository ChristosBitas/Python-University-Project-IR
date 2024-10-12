import pandas as pd
import spacy
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("_data/text_data.csv") # read csv file with project Guttemberg documents
nlp = spacy.load("en_core_web_sm") # load spacy and change max document length
nlp.max_length = np.inf

text_list = df.text.str.lower().values
tok_text=[] # for our tokenised corpus
#Tokenising using SpaCy:
for doc in tqdm(nlp.pipe(text_list, disable=["tagger", "parser","ner"])):
   tok = [t.text for t in doc if t.is_alpha]
   tok_text.append(tok)

bm25 = BM25Okapi(tok_text) # create a MB25 object
query = "rose" # query to use
tokenized_query = query.lower().split(" ") # preprocess the query
import time
results = bm25.get_top_n(tokenized_query, df.index, n=3) # get index of top n documents
print(results)

query_df = pd.DataFrame() # create a new dataframe based on query
query_df["doc"] = ["query"]
query_df["text"] = [query]
query_df

vec = TfidfVectorizer(stop_words="english", lowercase=True) # compute the tf idf dataframe of both corpus and query
data = vec.fit_transform(np.concatenate([df.text.values, query_df.text.values]))
features = vec.get_feature_names_out()
tf_idf_df = pd.DataFrame(data=data.toarray(), columns=features)
tf_idf_df
# calculate the cosine similarity and get top 3 documents (we take 4 and remove the first one)
pd.Series(cosine_similarity(tf_idf_df.values, tf_idf_df.values)[-1]).nlargest(n=4).index[1:].to_list()