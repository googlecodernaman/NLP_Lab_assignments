"""
NLP Assignment: Bag-of-Words, TF-IDF, and Word2Vec
"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

corpus = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document"
]

print("="*80)
print("CORPUS")
print("="*80)
for i, doc in enumerate(corpus):
    print(f"{i+1}. {doc}")

#Count Occurrence
print("\n" + "="*80)
print("BAG OF WORDS - COUNT OCCURRENCE")
print("="*80)
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(corpus)
print(f"\nVocabulary: {count_vectorizer.get_feature_names_out()}")
print(f"\nCount Matrix:\n{count_matrix.toarray()}")

#Normalized Count Occurrence
print("\n" + "="*80)
print("BAG OF WORDS - NORMALIZED COUNT OCCURRENCE")
print("="*80)
normalized_count = count_matrix.toarray() / count_matrix.sum(axis=1).reshape(-1, 1)
print(f"\nNormalized Count Matrix:\n{normalized_count}")

# TF-IDF
print("\n" + "="*80)
print("TF-IDF")
print("="*80)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
print(f"\nVocabulary: {tfidf_vectorizer.get_feature_names_out()}")
print(f"\nTF-IDF Matrix:\n{tfidf_matrix.toarray()}")

# Word2Vec
print("\n" + "="*80)
print("WORD2VEC EMBEDDINGS")
print("="*80)
# Tokenize corpus
tokenized_corpus = [doc.lower().split() for doc in corpus]
print(f"\nTokenized Corpus: {tokenized_corpus}")

# Train Word2Vec model
w2v_model = Word2Vec(sentences=tokenized_corpus, vector_size=10, window=2, min_count=1, workers=1)
print(f"\nVocabulary: {list(w2v_model.wv.index_to_key)}")

# Show embeddings
print("\nWord Embeddings:")
for word in w2v_model.wv.index_to_key:
    print(f"{word}: {w2v_model.wv[word]}")

print(f"\nExample - Embedding for 'document': {w2v_model.wv['document']}")

print(f"\nMost similar words to 'document': {w2v_model.wv.most_similar('document', topn=3)}")
