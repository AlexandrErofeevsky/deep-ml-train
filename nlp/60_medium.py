"""
Implement TF-IDF (Term Frequency-Inverse Document Frequency)
 - Medium
 - NLP

Task: Implement TF-IDF (Term Frequency-Inverse Document Frequency)
Your task is to implement a function that computes the TF-IDF scores for a query
against a given corpus of documents.

Function Signature
Write a function compute_tf_idf(corpus, query) that takes the following inputs:
    corpus: A list of documents, where each document is a list of words.
    query: A list of words for which you want to compute the TF-IDF scores.

Output
    The function should return a list of lists containing the TF-IDF scores for
    the query words in each document, rounded to five decimal places.

Important Considerations
    1. Handling Division by Zero:
    When implementing the Inverse Document Frequency (IDF) calculation, you must
    account for cases where a term does not appear in any document (df = 0).
    This can lead to division by zero in the standard IDF formula. Add smoothing
    (e.g., adding 1 to both numerator and denominator) to avoid such errors.

    2. Empty Corpus:
    Ensure your implementation gracefully handles the case of an empty corpus.
    If no documents are provided, your function should either raise an appropriate
    error or return an empty result. This will ensure the program remains robust
    and predictable.

Edge Cases:
 - Query terms not present in the corpus.
 - Documents with no words.
 - Extremely large or small values for term frequencies or document frequencies.

By addressing these considerations, your implementation will be robust and handle real-world scenarios effectively.

Example:
    Input:
        corpus = [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "chased", "the", "cat"],
            ["the", "bird", "flew", "over", "the", "mat"]
        ]
        query = ["cat"]

        print(compute_tf_idf(corpus, query))
    Output:
        [[0.21461], [0.25754], [0.0]]
Reasoning:
    The TF-IDF scores for the word "cat" in each document are computed and rounded
    to five decimal places.
"""
from collections import Counter

import numpy as np


def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    counters = [Counter(doc) for doc in corpus]

    def tf(t, d):
        return counters[d][t] / len(corpus[d])

    def idf(t):
        return np.log(
            (len(corpus) + 1) /
            (len([1 for text in counters if text[t] > 0]) + 1)
        ) + 1

    def tf_idf(t, d):
        return tf(t, d) * (idf(t))

    return [[tf_idf(word, i) for word in query] for i in range(len(corpus))]
