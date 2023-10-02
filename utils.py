import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


def fit_linear_regressoin(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    model = LinearRegression().fit(X, y)
    a = model.coef_[0][0]
    b = model.intercept_[0]
    return a, b


def plot_rank_freqs(sorted_words: list[tuple[str, int]]) -> None:
    freqs = [freq for _, freq in sorted_words]
    ranks = list(range(1, len(freqs)+1))
    plt.loglog(ranks, freqs, label='Log-Log Rank vs Frequency')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs)
    X = log_ranks.reshape(-1, 1)
    y = log_freqs.reshape(-1, 1)
    a, b = fit_linear_regressoin(X, y)
    plt.plot(ranks, 10**(a*log_ranks + b), label='Linear Estimation')
    plt.legend()
    plt.show();
    

def calc_tf_idf(documents: pd.Series, word_freq: Counter) -> tuple[dict, dict, dict]:
    tf_dict: dict[int, dict[str, int]] = {}
    # Calculate the term frequency for each document
    for i in range(len(documents)):
        words = documents[i]
        tf_dict[i] = {}
        for word in words:
            tf_dict[i][word] = tf_dict[i].get(word, 0) + 1

    sample = tf_dict[0]['april']
    print(f"Term - april's tf_score: {sample}")

    # add log-scale: 1+log10(tf)
    for d, freqs in tf_dict.items():
        for t, f in freqs.items():
            tf_dict[d][t] = 1 + np.log10(f)

    assert tf_dict[0]['april'] == 1+np.log10(sample)
    print(f"Term - april's tf_score (log-scale): {tf_dict[0]['april']}")

    idf_dict = {}
    for i in tqdm(range(len(documents))):
        words = documents[i]
        for word in set(words):
            idf_dict[word] = idf_dict.get(word, 0) + 1

    print(f"Term - april's idf_score (raw): {idf_dict['april']}")
    for word in idf_dict:
        idf_dict[word] = np.log10(len(documents) / idf_dict[word])

    print(f"Term - april's idf_score (log-scale): {idf_dict['april']}")

    # Calculate the TF-IDF values for each document
    tfidf_dict = {}
    for i in range(len(documents)):
        words = documents[i]
        tfidf_dict[i] = {}
        for word in set(words):
            tfidf_dict[i][word] = tf_dict[i][word] * idf_dict[word]
    return tf_dict, idf_dict, tfidf_dict

def normalize_tfidf(tfidf_matrix):
    """
    Normalize a TF-IDF matrix by row to have unit length.

    Parameters:
    tfidf_matrix (numpy.ndarray): Matrix of TF-IDF values.

    Returns:
    numpy.ndarray: Normalized TF-IDF matrix.
    """
    return normalize(tfidf_matrix, norm='l2', axis=1)

def process_query(query):
    """
    Preprocesses the query, calculates its TF-IDF representation, and normalizes the resulting vector.
    """
    # Tokenization, lemmatization and lowercase transformation
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(query)]
    stemmer = PorterStemmer()
    normalized = [stemmer.stem(word) for word in tokens]
    text = " ".join(normalized)
    
    # Calculate the term frequency (TF) for the query
    tfidf_vectorizer_query = TfidfVectorizer(use_idf=True, vocabulary=TfidfVectorizer(use_idf=True).vocabulary_)
    tfidf_matrix = tfidf_vectorizer_query.fit_transform([text])
    tf = tfidf_matrix.toarray()
    tf1 = np.float16(tf)
    idf = tfidf_vectorizer_query.idf_
    idf1 = np.float16(idf)

    # Calculate the TF-IDF for the query
    tf_idf_query = tf1 * idf1

    # Normalize the query TF-IDF vector
    tf_idf_query_norm = np.float16(normalize_tfidf(tf_idf_query))
    
    return tf_idf_query_norm[0]

def search(query, k, tf_idf_norm, text_df):
    """
    Returns the top-k most relevant documents for the given query using cosine similarity.
    """
    # Preprocess and normalize the query
    tf_idf_query_norm = process_query(query)
    
    # Calculate the cosine similarity between the query and each document
    similarities = cosine_similarity(tf_idf_query_norm.reshape(1,-1), tf_idf_norm)[0].astype(np.float16)
    
    # Sort the documents by their similarity to the query and return the top-k results
    top_k_indices = similarities.argsort()[::-1][:k]
    top_k_similarities = similarities[top_k_indices]
    top_k_documents = text_df.iloc[top_k_indices]['normalized'].values.tolist()
    
    return top_k_documents, top_k_similarities